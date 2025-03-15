import json
from flask import Flask, request, jsonify, Response, render_template
import asyncio
import time
import re
from typing import List, Tuple, Optional, Dict

from playwright.async_api import async_playwright
import uuid
from concurrent.futures import ThreadPoolExecutor
import asyncio  # Import asyncio.Queue

app = Flask(__name__, static_folder='static', template_folder='.')

CONCURRENT_LIMIT = 20
search_cache: Dict[str, str] = {}
active_search_tasks: Dict[str, asyncio.Event] = {}
sem = asyncio.Semaphore(CONCURRENT_LIMIT)
executor = ThreadPoolExecutor(max_workers=4)
# _stream_queue: asyncio.Queue = asyncio.Queue()

def is_subsequence(small: List[str], big: List[str]) -> bool:
    it = iter(big)
    return all(word in it for word in small)

def normalize_text(text: str) -> str:
    text = text.replace('-', '').replace(',', '')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text

async def wait_for_results_update(page) -> None:
    await page.wait_for_function(
        "document.querySelector('span.page-results') && document.querySelector('span.page-results').textContent.trim() !== ''",
        timeout=0
    )

async def binary_search_partial(term: str, page, base_url: str, cancel_event: asyncio.Event) -> Optional[str]:
    words = term.split()
    lo, hi = 1, len(words)
    best: Optional[str] = None
    while lo <= hi:
        if cancel_event.is_set():
            return None
        mid = (lo + hi) // 2
        prefix = " ".join(words[:mid])
        print(f"Term {term}: binary_search_partial - Trying prefix: '{prefix}'") # Debug: Prefix being tried
        await page.goto(base_url, wait_until="networkidle", timeout=0)
        await page.wait_for_selector("div.main-search input.search-term", timeout=30000)
        await page.fill("div.main-search input.search-term", prefix)
        await page.press("div.main-search input.search-term", "Enter")
        try:
            await wait_for_results_update(page)
            print(f"Term {term}: binary_search_partial - wait_for_results_update completed for prefix: '{prefix}'") # Debug: wait_for_results
        except asyncio.TimeoutError:
            partial_content = ""
            print(f"Term {term}: binary_search_partial - TimeoutError in wait_for_results_update for prefix: '{prefix}'") # Debug: Timeout
        else:
            partial_content = (await page.text_content("span.page-results")) or ""
        if partial_content and "Displaying" in partial_content:
            print(f"Term {term}: binary_search_partial - 'Displaying' found for prefix: '{prefix}', content: '{partial_content[:100]}...'") # Debug: Displaying found
            best = prefix
            lo = mid + 1
        else:
            print(f"Term {term}: binary_search_partial - 'Displaying' NOT found for prefix: '{prefix}', content: '{partial_content[:100]}...'") # Debug: Displaying not found
            hi = mid - 1
    print(f"Term {term}: binary_search_partial - Best prefix found: '{best}'") # Debug: Best prefix
    return best


async def search_term(term: str, base_url: str, context, cancel_event: asyncio.Event, semaphore: asyncio.Semaphore, progress_callback, stream_queue) -> Tuple[str, str]: # Removed current_loop
    print(f"Searching for term: {term}")
    if cancel_event.is_set():
        print(f"Term {term}: Cancelled early")
        return term, "Cancelled"
    if term in search_cache:
        print(f"Term {term}: Cache hit")
        return term, search_cache[term]
    async with semaphore:
        page = await context.new_page()
        try:
            await page.goto(base_url, wait_until="networkidle", timeout=0)
            await page.wait_for_selector("div.main-search input.search-term", timeout=30000)
            await page.fill("div.main-search input.search-term", term)
            await page.press("div.main-search input.search-term", "Enter")
            try:
                await wait_for_results_update(page)
            except asyncio.TimeoutError:
                content = ""
            else:
                content = (await page.text_content("span.page-results")) or ""

            initial_result_type = ""
            full_match_prefix = "Displaying search results for:"

            partial: Optional[str] = None

            print(f"Term {term}: Initial content (page-results): {content[:100]}")

            if content and full_match_prefix in content:
                displayed_term_match = re.search(rf"{re.escape(full_match_prefix)}\s*\"(.+?)\"", content)
                if displayed_term_match:
                    displayed_term = displayed_term_match.group(1).strip()
                    if normalize_text(term) == normalize_text(displayed_term):
                        initial_result_type = "full_match_prefix"
                    else:
                        initial_result_type = "larger_description_prefix"
                else:
                    initial_result_type = "larger_description_prefix_fail"
            elif content and "Displaying" in content:
                initial_result_type = "larger_description_general"
            else:
                print(f"Term {term}: No 'Displaying' in initial content, calling binary_search_partial")
                partial = await binary_search_partial(term, page, base_url, cancel_event)
                print(f"Term {term}: Partial after binary search: {partial}")
                if partial:
                    print(f"Term {term}: Partial prefix found: '{partial}', proceeding with partial search.") # Debug: Partial found
                    await page.goto(base_url, wait_until="networkidle", timeout=0)
                    await page.wait_for_selector("div.main-search input.search-term", timeout=30000)
                    await page.fill("div.main-search input.search-term", partial)
                    await page.press("div.main-search input.search-term", "Enter")
                    try:
                        await wait_for_results_update(page)
                        print(f"Term {term}: search_term - wait_for_results_update after partial search completed.") # Debug: wait_for_results after partial
                    except asyncio.TimeoutError:
                        print(f"Term {term}: search_term - TimeoutError in wait_for_results_update after partial search.") # Debug: Timeout after partial

                    description_cells = await page.query_selector_all("td[data-column='description']")
                    print(f"Term {term}: Description cells found after partial search (after wait_for_results_update in search_term): {len(description_cells)}")

                    normalized_term = normalize_text(term)
                    normalized_partial = normalize_text(partial)
                    term_id_number = "Not found"  # Initialize term_id_number *outside* if block
                    specific_description_example = "Description example not found" # Initialize example description

                    if description_cells:
                        # --- Robust Term ID Extraction ---
                        for cell in description_cells:
                            parent_row = await cell.evaluate_handle("node => node.parentElement")
                            term_id_element = await parent_row.query_selector("a.view-record")
                            if term_id_element:
                                current_term_id_number = (await term_id_element.text_content()).strip()
                                if term_id_number == "Not found": # Take the first found term ID if not already found
                                    term_id_number = current_term_id_number
                                print(f"Term {term}: Found term_id_number: {term_id_number} within description cell row (search_term).")

                                # --- Fetch Description Example ---
                                if term_id_number != "Not found" and current_term_id_number == term_id_number: # Ensure we are using the correct row
                                    example_description_cell = await parent_row.query_selector("td[data-column='description']")
                                    if example_description_cell:
                                        specific_description_example = (await example_description_cell.text_content()).strip()
                                        print(f"Term {term}: Found example description: '{specific_description_example}' for Term ID: {term_id_number}")
                                # --- End Fetch Description Example ---
                                break # Take the first row's term ID and example description
                        # --- Robust Term ID Extraction End ---
                    else:
                        print(f"Term {term}: No description cells found after partial search (search_term), even after wait_for_results_update.")


                    if normalized_partial in normalized_term:
                        initial_result_type = "acceptable_specific"
                        print(f"Term {term}: Categorized as acceptable_specific (search_term).")
                        result = f"Acceptable Specific Description (Term ID: {term_id_number or 'Not found'}) - {specific_description_example}" # Include example description in result
                        search_cache[term] = result
                        print(f"Term {term}: Final result: {result} (Acceptable Specific)")
                        return term, result # Explicit return here for acceptable_specific - IMPORTANT!

                    elif description_cells: # If partial prefix is NOT a substring, but we *do* have description cells, then it's potentially vague
                        initial_result_type = "vague_description"
                        # Get the first description cell's text as an example
                        description_text = (await description_cells[0].text_content()).strip() if description_cells else "Description not found"
                        parent_row = await description_cells[0].evaluate_handle("node => node.parentElement") if description_cells else None
                        term_id_element = await parent_row.query_selector("a.view-record") if parent_row else None
                        term_id_number = (await term_id_element.text_content()).strip() if term_id_element else "Not found"
                        print(f"Term {term}: Categorized as vague_description because partial prefix is NOT substring but descriptions found.")

                    else:
                        initial_result_type = "no_match"
                        print(f"Term {term}: No description cells found after partial search, setting to no_match")

                else:
                    initial_result_type = "no_match"
                    print(f"Term {term}: No partial prefix found, setting to no_match")


            print(f"Term {term}: Initial result type: {initial_result_type}")

            description_text = "Not found"
            term_id_number = "Not found" # Redundant initialization here, but harmless
            is_deleted_description = False
            found_full_description_match = False
            found_in_description = False
            view_record_link = await page.query_selector("a.view-record")
            matched_cell_text = ""

            if initial_result_type not in ["vague_description", "no_match", "acceptable_specific"]:
                if view_record_link:
                    term_id_number = (await view_record_link.text_content()).strip()

                description_cells = await page.query_selector_all("td[data-column='description']")
                print(f"Term {term}: Description cells (full/wordy check) found: {len(description_cells)}")


                for cell in description_cells:
                    cell_text = (await cell.text_content()).strip()
                    normalized_cell_text = normalize_text(cell_text)
                    normalized_term = normalize_text(term)
                    print(f"Term {term}: Checking cell text for full match/wordy: '{cell_text}'")

                    if normalized_term == normalized_cell_text:
                        found_full_description_match = True
                        found_in_description = True
                        matched_cell_text = cell_text
                        parent_row = await cell.evaluate_handle("node => node.parentElement")
                        notes_element = await parent_row.query_selector("td[data-column='notes']")
                        if notes_element:
                            notes_text = (await notes_element.text_content()).strip()
                            if re.search(r"deleted", normalize_text(notes_text)):
                                is_deleted_description = True
                                break
                        break

                    elif normalized_term in normalized_cell_text:
                        found_in_description = True
                        parent_row = await cell.evaluate_handle("node => node.parentElement")
                        notes_element = await parent_row.query_selector("td[data-column='notes']")
                        if notes_element:
                            notes_text = (await notes_element.text_content()).strip()
                            if re.search(r"deleted", normalize_text(notes_text)):
                                is_deleted_description = True
                                break


            print(f"Term {term}: found_full_description_match: {found_full_description_match}, found_in_description: {found_in_description}, is_deleted_description: {is_deleted_description}")

            if initial_result_type == "vague_description":
                result = f"Vague Description (Example - {description_text} - Term ID: {term_id_number})"
            elif is_deleted_description:
                result = f"Deleted Description (Term ID: {term_id_number})"
            elif initial_result_type == "acceptable_specific":
                result = f"Acceptable Specific Description (Term ID: {term_id_number or 'Not found'}) - {specific_description_example}" # Use example description here as well for consistency, although it should have been set already
            elif found_full_description_match:
                initial_result_type = "full_match"
                normalized_term = normalize_text(term)
                normalized_matched_cell_text = normalize_text(matched_cell_text)
                term_word_count = len(normalized_term.split())
                cell_word_count = len(normalized_matched_cell_text.split())

                if term_word_count > cell_word_count:
                    initial_result_type = "acceptable_wordy"
                    result = f"Acceptable but Wordy Description (Full match found, but applicant's description is more detailed - Term ID: {term_id_number})"
                else:
                    result = f"Full Match Found (Term ID: {term_id_number})"
            elif initial_result_type == "no_match":
                result = "No Match Found - Further Review Needed"
            elif view_record_link:
                description_element = await page.query_selector("td[data-column='description']")
                if description_element:
                    dt = await description_element.text_content()
                    if dt:
                        description_text = dt.strip()
                result = f"Vague Description (Example - {description_text} - Term ID: {term_id_number})"
            else:
                result = "No Match Found - Further Review Needed (Description/Term ID not found)"

            search_cache[term] = result
            print(f"Term {term}: Final result: {result}")
            return term, result
        finally:
            await page.close()

async def run_searches(terms: List[str], base_url: str, progress_callback, result_callback, search_id, app_context, stream_queue): # Removed current_loop
    cancel_event = asyncio.Event()
    active_search_tasks[search_id] = cancel_event
    results = {}
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        tasks = []
        for term in terms:
            task = asyncio.create_task(search_term(term, base_url, context, cancel_event, sem, lambda p: progress_callback(p), stream_queue)) # Removed current_loop
            tasks.append(task)

        completed_count = 0
        for task in asyncio.as_completed(tasks):
            try:
                term, result_string = await task  # Get result string from search_term
                results[term] = result_string  # Store result string in results dict
                result_type = get_result_type(result_string)  # Function to determine result type from string
                result_callback({"type": "result", "term": term, "message": generate_result_html(term, result_string), "result_type": result_type}, app_context, stream_queue)  # Removed current_loop
            except asyncio.CancelledError:
                term = "Cancelled"
                result_string = "Cancelled"
                results[term] = result_string
                result_type = "cancelled"  # Set result type for cancelled
                result_callback({"type": "result", "term": term, "message": generate_result_html(term, result_string), "result_type": result_type}, app_context, stream_queue)  # Removed current_loop
            except Exception as e:
                term = "Error"  # Or keep the original term if available
                result_string = f"Error: {str(e)}"
                results[term] = result_string
                result_type = "error"  # Set result type for error
                result_callback({"type": "result", "term": term, "message": generate_result_html(term, result_string), "result_type": result_type}, app_context, stream_queue)  # Removed current_loop
                continue

            completed_count += 1
            progress_percentage = int((completed_count / len(terms)) * 100)
            progress_callback({"type": "progress", "percentage": progress_percentage}, app_context, stream_queue) # Removed current_loop

        await context.close()
        await browser.close()
    try:
        await stream_queue.put(None)  # Add sentinel value to signal end of stream - use local stream_queue
    except:
        print("Failed to add sentinel value to queue")
    del active_search_tasks[search_id]
    return results, cancel_event

def get_result_type(result_string: str) -> str:  # Helper function to extract result type from string
    if "No Match Found" in result_string:
        return "needs_further_review" # Renamed from no_match
    elif "Full Match Found" in result_string:
        return "full_match"
    elif "Acceptable but Wordy Description" in result_string: # Renamed from partial
        return "acceptable_wordy"
    elif "Vague Description" in result_string: # Renamed from larger_description
        return "vague_description"
    elif "Deleted Description" in result_string: # Renamed from deleted_description
        return "deleted_description"
    elif "Acceptable Specific Description" in result_string: # Added for acceptable specific
        return "acceptable_specific"
    elif "Cancelled" in result_string:
        return "cancelled"
    elif "Error:" in result_string:
        return "error"
    else:
        return "unknown"  # Default type if none of the above match

@app.route('/search', methods=['POST'])
def search():  # Changed to def (synchronous)
    terms_string = request.get_json().get('terms', '') # Capture request data here, *outside* generator
    terms = [term.strip() for term in terms_string.split(';') if term.strip()]
    if not terms:
        return jsonify({"error": "No search terms provided"}), 400

    search_id = str(uuid.uuid4())
    app_context = app.app_context() # Get app context here
    stream_queue_local = asyncio.Queue() # Create queue here


    def progress_callback(update, context, stream_queue):  # Synchronous callback, accepts app_context and stream_queue, removed current_loop
        try:
            with context: # Push context
                loop = asyncio.get_event_loop() # Get the *current* loop here in callback
                asyncio.run_coroutine_threadsafe(stream_queue.put(f"{jsonify(update).get_data(as_text=True)}\n"), loop) # Use run_coroutine_threadsafe, use local stream_queue, get loop in callback
        except asyncio.QueueFull:
            print("Warning: _stream_queue is full, progress message dropped.")

    def result_callback(result_update, context, stream_queue):  # Synchronous callback, accepts app_context and stream_queue, removed current_loop
        try:
            with context: # Push context
                loop = asyncio.get_event_loop() # Get the *current* loop here in callback
                asyncio.run_coroutine_threadsafe(stream_queue.put(f"{jsonify(result_update).get_data(as_text=True)}\n"), loop) # Use run_coroutine_threadsafe, use local stream_queue, get loop in callback
        except asyncio.QueueFull:
            print("Warning: _stream_queue is full, result message dropped.")


    async def stream_results_async(search_terms, stream_queue):  # Renamed to stream_results_async, keep as async generator, accept stream_queue, removed current_loop
        global search_cache
        global search_start_time
        # global _stream_queue # No longer using global queue

        search_cache = {}
        search_start_time = time.time()
        # _stream_queue.queue.clear() # No longer needed for asyncio.Queue, clear differently if needed

        try:
            yield (json.dumps({'type': 'progress', 'percentage': 0}) + "\n").encode('utf-8')

            base_url = "https://idm-tmng.uspto.gov/id-master-list-public.html"
            loop = asyncio.get_event_loop() # Get the *current* loop here in async function
            asyncio.create_task(run_searches(search_terms, base_url, progress_callback, result_callback, search_id, app_context, stream_queue)) # Pass app_context and stream_queue to run_searches, removed loop

            while True: # Loop to get messages from stream_queue
                message_str = await stream_queue.get() # Asynchronously get message from queue - use local stream_queue
                if message_str is None: # Add a sentinel value to signal end of stream if needed in run_searches
                    break
                yield message_str.encode('utf-8') # Yield the message as bytes
                stream_queue.task_done() # Signal task completion to queue - use local stream_queue

            final_results_html = generate_final_results_html(search_cache)
            elapsed_time = time.time() - search_start_time

            yield (json.dumps({'type': 'final', 'message': final_results_html}) + "\n").encode('utf-8')
            yield (json.dumps({'type': 'time', 'message': f'{elapsed_time:.2f} seconds'}) + "\n").encode('utf-8')
            yield b''
        finally:
            pass


    def stream_results_sync_wrapper(search_terms): # Synchronous wrapper for Flask Response
        loop = asyncio.new_event_loop() # Create a new event loop for the sync context
        asyncio.set_event_loop(loop) # Set it as the current loop
        try:
            # Run the async generator and collect results in a synchronous generator
            stream_queue_wrapper = asyncio.Queue() # Create queue here for wrapper - not needed actually.
            result_generator = stream_results_async(search_terms, stream_queue_local) # Get the async generator, pass local queue, removed loop
            while True:
                try:
                    item_bytes = loop.run_until_complete(result_generator.__anext__()) # Await the next item from async generator
                    yield item_bytes # Yield bytes to Flask Response
                except StopAsyncIteration: # Async generator is exhausted
                    break
        finally:
            loop.close()

    return Response(stream_results_sync_wrapper(terms), mimetype='text/plain') # Use the sync wrapper in Response


@app.route('/cancel', methods=['POST'])
def cancel():
    search_id = request.json.get('search_id')
    if not search_id:
        return jsonify({"error": "No search ID provided for cancellation"}), 400

    cancel_event = active_search_tasks.get(search_id)
    if cancel_event:
        cancel_event.set()
        return jsonify({"status": "cancelled", "search_id": search_id}), 200
    else:
        return jsonify({"status": "search_id_not_found", "search_id": search_id}), 404


def generate_final_results_html(results):
    def capitalize_term(term_string):
        return term_string.strip().capitalize()

    needs_further_review_results = [] # Renamed from no_results
    acceptable_wordy_results = [] # Renamed from partial_results
    full_matches = []
    vague_description_results = [] # Renamed from larger_description_results
    deleted_descriptions_results = []
    acceptable_specific_results = [] 

    for term, status in results.items():
        if status == "No Match Found - Further Review Needed": # Updated status string
            needs_further_review_results.append((term, status)) # Updated list name
        elif status.startswith("Full Match Found"):
            full_matches.append((term, status))
        elif status.startswith("Acceptable but Wordy Description"): # Updated status string
            acceptable_wordy_results.append((term, status)) # Updated list name
        elif status.startswith("Vague Description"): # Updated status string
            vague_description_results.append((term, status)) # Updated list name
        elif status.startswith("Deleted Description"): # Updated status string
            deleted_descriptions_results.append((term, status))
        elif status.startswith("Acceptable Specific Description"): 
            acceptable_specific_results.append((term, status)) # POPULATE THE LIST

    html_final = "" # Initialize as empty string, no HTML wrapper

    if needs_further_review_results: # Updated list name
        html_final += "<h2>Descriptions Needing Further Review</h2><ol>" # Updated heading
        for term, _ in needs_further_review_results: # Updated list name
            html_final += f"<li><span class='result-term'>{capitalize_term(term)}:</span> No Match Found - Further Review Needed</li>" # Updated message
        html_final += "</ol>"
    if acceptable_wordy_results: # Updated list name
        html_final += "<h2>Acceptable but Wordy Descriptions</h2><ol>" # Updated heading
        for term, status in acceptable_wordy_results: # Updated list name
            m = re.search(r"Acceptable but Wordy Description \(Full match found, but applicant's description is more detailed - Term ID: (.+?)\)", status) # Updated regex
            term_id_wordy = m.group(1) if m else "Not found"
            html_final += (
                f"<li><span class='result-term'>{capitalize_term(term)}:</span> "
                f"<span class='wordy-description'>Acceptable but Wordy Description</span> (Term ID: {term_id_wordy}). " # Updated class and message
                f"Description is acceptable but could be more concise.</li>" # Updated message
            )
        html_final += "</ol>"
    if vague_description_results: # Updated list name
        html_final += "<h2>Vague Descriptions</h2><ol>" # Updated heading
        for term, status in vague_description_results: # Updated list name
            m = re.search(r"Vague Description \(Example - (.+?) - Term ID: (.+?)\)", status) # Updated regex
            description_text = m.group(1) if m else "Description not found"
            term_id_vague = m.group(2) if m else "Not found"
            html_final += f"<li><span class='result-term'>{capitalize_term(term)}:</span> <span class='vague-description'>Vague Description</span> (Example - <span class='example-description'>{description_text}</span> - Term ID: {term_id_vague}). Description is likely too broad or incomplete and needs clarification.</li>" # Updated class and message
        html_final += "</ol>"
    if deleted_descriptions_results:
        html_final += "<h2>Deleted Descriptions (Unacceptable)</h2><ol>" # Updated heading
        for term, status in deleted_descriptions_results:
            m = re.search(r"Deleted Description \(Term ID: (.+?)\)", status) # Updated regex
            term_id_deleted = m.group(1) if m else "Not found"
            html_final += f"<li><span class='result-term'>{capitalize_term(term)}:</span> <span class='deleted-description'>{term}</span> (Term ID: <span class='term-id'>{term_id_deleted}</span>) - <span class='deleted-description'>Deleted Description - Unacceptable.</span></li>" # Updated message, term only once, term ID styled
        html_final += "</ol>"
    if full_matches:
        html_final += "<h2>Full Match Found (Acceptable)</h2><ol>" # Updated heading
        for term, status in full_matches:
            m = re.search(r"Full Match Found \(Term ID: (.+?)\)", status) # Updated regex
            term_id_full = m.group(1) if m else "Not found"
            html_final += f"<li><span class='result-term'>{capitalize_term(term)}:</span> Full Match Found (Term ID: <span class='term-id'>{term_id_full}</span>) - <span class='acceptable-description'>Acceptable Description.</span></li>" # Updated message, term ID styled, status class
        html_final += "</ol>"
    if acceptable_specific_results: 
        html_final += "<h2>Acceptable Specific Descriptions</h2><ol>" # New Heading
        for term, status in acceptable_specific_results: # Iterate new list
            m = re.search(r"Acceptable Specific Description \(Term ID: (.+?)\) - (.+)", status) # Updated regex to capture description
            term_id_specific = m.group(1) if m else "Not found"
            specific_description_example_final = m.group(2) if m and m.lastindex >= 2 else "Description example not found" # Extract description
            html_final += f"<li><span class='result-term'>{capitalize_term(term)}:</span> <span class='acceptable-specific-description'>Acceptable Specific Description</span>.<br> <span class='example-intro'>For example:</span> <span class='example-description'><i>{specific_description_example_final}</i></span> (<span class='term-id'>Term ID: {term_id_specific}</span>). This description is a specific instance of a broader acceptable category.</li>" # Updated message, added example description, formatted example and term ID, intro text
        html_final += "</ol>"
    # No more </body></html> wrapper

    return html_final

def generate_result_html(term, status):
    term_display = f"<b>{term.strip().capitalize()}:</b> "
    if status == "No Match Found - Further Review Needed": # Updated status string
        status_display = f"<span class='needs-review-description'>No Match Found - Further Review Needed</span>" # Updated color, class added
        color = "#333"
    elif status.startswith("Full Match Found"):
        status_display = f"<span class='acceptable-description'>Full Match Found - Acceptable</span>" # Updated message, class added
        color = "#333"
    elif status.startswith("Acceptable but Wordy Description"): # Updated status string
        status_display = f"<span class='wordy-description'>{status}</span>" # Updated color and style, class added
        color = "#333"
    elif status.startswith("Vague Description"): # Updated status string
        status_display = f"<span class='vague-description'>{status}</span>" # Updated color, class added
        color = "#333"
    elif status.startswith("Deleted Description"): # Updated status string
        status_display = f"<span class='deleted-description'>{status} - Unacceptable</span>" # Updated message, class added
        color = "#333"
    elif status.startswith("Acceptable Specific Description"): # Updated condition
        m = re.search(r"Acceptable Specific Description \(Term ID: (.+?)\) - (.+)", status) # Updated regex to capture description
        term_id_specific_inline = m.group(1) if m else "Not found"
        specific_description_example_inline = m.group(2) if m and m.lastindex >= 2 else "Description example not found" # Extract description
        status_display = f"<span class='acceptable-specific-description'>Acceptable Specific Description</span>.<br> <span class='example-intro'>For example:</span> <span class='example-description'><i>{specific_description_example_inline}</i></span> (<span class='term-id'>Term ID: {term_id_specific_inline}</span>). This description is a specific instance of a broader acceptable category." # Modified line, added example description, formatted example and term ID, intro text, classes added
        color = "#333"
    elif status == "Cancelled":
        status_display = "<span class='cancelled-description'>Cancelled</span>" # Class added
        color = "#333"
    elif status.startswith("Error:"):
        status_display = f"<span class='error-description'>{status}</span>" # Class added
        color = "#333"
    else:
        status_display = status
        color = "#333"

    return f"<p style='color: {color}; font-size: 25px;'>{term_display}{status_display}</p>"


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)