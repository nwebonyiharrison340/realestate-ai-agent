document.addEventListener("DOMContentLoaded", () => {
  const chatWidget = document.getElementById("chat-widget");
  const chatToggleBtn = document.getElementById("chat-toggle-btn");
  const sendBtn = document.getElementById("send-btn");
  const userInput = document.getElementById("user-input");
  const chatBody = document.getElementById("chat-body");

  // Toggle Chat Widget
  chatToggleBtn.addEventListener("click", () => {
    chatWidget.style.display = chatWidget.style.display === "flex" ? "none" : "flex";
  });

  // Send Message
  const sendMessage = async () => {
    const message = userInput.value.trim();
    if (!message) return;

    appendMessage(message, "user-message");
    userInput.value = "";

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const data = await response.json();
      appendMessage(data.response, "bot-message");
    } catch {
      appendMessage("⚠️ Error connecting to server.", "bot-message");
    }
  };

  // Handle Send Button and Enter Key
  sendBtn.addEventListener("click", sendMessage);
  userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
  });

  // Append messages dynamically
  function appendMessage(text, className) {
    const msg = document.createElement("div");
    msg.className = `message ${className}`;
    msg.textContent = text;
    chatBody.appendChild(msg);
    chatBody.scrollTop = chatBody.scrollHeight;
  }
});
