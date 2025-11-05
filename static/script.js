document.getElementById("send-btn").addEventListener("click", sendMessage);
document.getElementById("user-input").addEventListener("keypress", function(e) {
    if (e.key === "Enter") sendMessage();
});

async function sendMessage() {
    const input = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const userMessage = input.value.trim();
    if (!userMessage) return;

    // Display user message
    chatBox.innerHTML += `<div class="text-right mb-2"><span class="bg-blue-500 text-white px-3 py-1 rounded-lg">${userMessage}</span></div>`;
    input.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userMessage })
        });

        const data = await response.json();
        const botMessage = data.response || "Sorry, I couldn’t process that.";

        // Display AI message
        chatBox.innerHTML += `<div class="text-left mb-2"><span class="bg-gray-200 px-3 py-1 rounded-lg">${botMessage}</span></div>`;
        chatBox.scrollTop = chatBox.scrollHeight;

    } catch (error) {
        chatBox.innerHTML += `<div class="text-left mb-2 text-red-600">Error: Could not connect to the server.</div>`;
    }
}
