console.log("SCRIPT.JS FILE LOADED AND RUNNING!"); // <--- ADD THIS LINE at the VERY TOP
document.addEventListener('DOMContentLoaded', () => {
    const searchTermsInput = document.getElementById('searchTerms');
    const searchButton = document.getElementById('searchButton');
    const cancelButton = document.getElementById('cancelButton');
    const progressBarFill = document.getElementById('progressBarFill');
    const progressPercent = document.getElementById('progressPercent');
    const outputDiv = document.getElementById('output');
    const hintLabel = document.getElementById('hintLabel');

    let searchInProgress = false;
    let currentSearchId = null; // Variable to store the current search ID

    // Function to update progress bar
    function updateProgressBar(percentage) {
        progressBarFill.style.width = `${percentage}%`;
        progressPercent.textContent = `${percentage}%`;
    }

    // Function to display message in output area (now handles HTML directly)
    function displayMessage(messageHtml) {
        outputDiv.innerHTML += messageHtml; // Append HTML directly
    }

    // Function to clear output area
    function clearOutputArea() {
        outputDiv.innerHTML = ""; // Clear HTML content
    }

    // Function to handle search request
    async function performSearch() {
        if (searchInProgress) return;
        searchInProgress = true;
        currentSearchId = generateSearchId(); // Generate a unique search ID

        const searchTerms = searchTermsInput.value.trim();
        if (!searchTerms) {
            displayMessage("<p style='color: red; font-size: 25px;'>Please enter search terms.</p>"); // Display error in red
            searchInProgress = false;
            currentSearchId = null;
            return;
        }

        clearOutputArea(); // Clear previous results
        displayMessage("<p style='color: #333; font-size: 25px;'>Searching...</p>"); // Indicate searching
        searchButton.disabled = true;
        cancelButton.disabled = false;
        updateProgressBar(0);

        try {
            const response = await fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ terms: searchTerms, search_id: currentSearchId }) // Send search terms and search_id
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const textDecoder = new TextDecoder();
            let partialData = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    break;
                }
                partialData += textDecoder.decode(value);

                console.log("Partial data received:", partialData); // <--- ADDED LOG

                const lines = partialData.split('\n');
                console.log("Lines after split:", lines); // <--- ADDED LOG
                partialData = lines.pop() || ''; // Keep the last incomplete line

                for (const line of lines) {
                    if (line) {
                        try {
                            console.log("Attempting to parse line:", line); // <--- ADDED LOG BEFORE PARSE
                            const result = JSON.parse(line);
                            console.log("Parsed JSON:", result);
                            console.log("Received JSON:", result); // Debug log

                            if (result.type === 'progress') {
                                updateProgressBar(result.percentage);
                            } else if (result.type === 'result') {
                                console.log("ENTERING RESULT TYPE BLOCK:", result); // <--- ADDED LOG - Check if outer 'result' block is reached

                                displayMessage(result.message); // Display individual result HTML - CORRECT LINE

                                console.log("result.result_type VALUE:", result.result_type); // <--- ADDED LOG - Check result.result_type value

                                if (result.result_type === 'acceptable_specific') {
                                    console.log("ACCEPTABLE SPECIFIC RESULT RECEIVED:", result); // <--- ADDED CONSOLE.LOG HERE
                                } else if (result.result_type === 'vague_description') {
                                    console.log("VAGUE DESCRIPTION RESULT RECEIVED:", result);
                                } // ... (handle other result types similarly) ...

                            } else if (result.type === 'final') {
                                outputDiv.innerHTML = result.message; // Set final HTML results
                            } else if (result.type === 'time') {
                                document.title = `USPTO Search Checker - Search time: ${result.message}`;
                                searchButton.disabled = false;
                                cancelButton.disabled = true;
                                searchInProgress = false;
                                currentSearchId = null; // Clear search ID after completion
                            } else if (result.type === 'script') {
                                console.log("Executing script from server:", result.content); // Debug log
                                let script = document.createElement('script');
                                script.text = result.content;
                                document.body.appendChild(script);
                                script.remove(); // Clean up after execution
                            }
                        } catch (parseError) {
                            console.error("Error parsing JSON line:", line, parseError);
                        }
                    }
                }
            }

        } catch (error) {
            console.error("Search error:", error);
            displayMessage(`<p style='color: red; font-size: 25px;'>Error: ${error.message}</p>`); // Display error in red
            searchButton.disabled = false;
            cancelButton.disabled = true;
            searchInProgress = false;
            currentSearchId = null; // Clear search ID on error
        }
    }

    // Function to handle cancel request
    async function cancelSearch() {
        if (!searchInProgress || !currentSearchId) return; // Check for search in progress and search ID

        try {
            const response = await fetch('/cancel', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ search_id: currentSearchId }) // Send search_id for cancel
            });

            if (!response.ok) {
                throw new Error(`HTTP error during cancel! status: ${response.status}`);
            }

            displayMessage("<p style='color: grey; font-size: 25px;'>Search cancelled by user.</p>"); // Display cancelled message
            searchButton.disabled = false;
            cancelButton.disabled = true;
            searchInProgress = false;
            currentSearchId = null; // Clear search ID after cancellation

        } catch (error) {
            console.error("Cancel error:", error);
            displayMessage(`<p style='color: red; font-size: 25px;'>Error cancelling search: ${error.message}</p>`); // Display error in red
            searchButton.disabled = false;
            cancelButton.disabled = true;
            searchInProgress = false;
            currentSearchId = null; // Clear search ID on cancel error
        }
    }

    // Function to generate a unique search ID (UUID-like)
    function generateSearchId() {
        return crypto.randomUUID ? crypto.randomUUID() : ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
            (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
        );
    }

    // Event listeners
    searchButton.addEventListener('click', performSearch);
    cancelButton.addEventListener('click', cancelSearch);

    searchTermsInput.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.ctrlKey) {
            event.preventDefault();
            performSearch();
        } else if (event.key === 'Escape') {
            cancelSearch();
        }
    });

    hintLabel.textContent = "Hint: Press Enter to search, Esc to cancel search, Ctrl+Enter for a new line.";

});