document.addEventListener("DOMContentLoaded", () => {
  const chatWidget = document.getElementById("chat-widget");
  const chatToggleBtn = document.getElementById("chat-toggle-btn");
  const sendBtn = document.getElementById("send-btn");
  const userInput = document.getElementById("user-input");
  const chatBody = document.getElementById("chat-body");
  const typingIndicator = document.getElementById("typing-indicator");

  // 🟢 Toggle Chat Widget visibility
  chatToggleBtn.addEventListener("click", () => {
    chatWidget.style.display = chatWidget.style.display === "flex" ? "none" : "flex";
  });

  // ✉️ Send message
  async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    appendMessage(message, "user-message");
    userInput.value = "";
    showTypingIndicator(); // show "Tinah is typing..."

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });

      const data = await response.json();
      hideTypingIndicator();

      // render formatted text (properties, blogs, etc.)
      appendMessage(formatBotMessage(data.response), "bot-message");
    } catch (err) {
      hideTypingIndicator();
      appendMessage("⚠️ Error connecting to the server.", "bot-message");
      console.error("Chat error:", err);
    }
  }

  // 🖱️ Handle Send button & Enter key
  sendBtn.addEventListener("click", sendMessage);
  userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
  });

  // 🧠 Append message to chat body
  function appendMessage(content, className) {
    const msg = document.createElement("div");
    msg.className = `message ${className}`;
    msg.innerHTML = content; // use HTML to allow formatting
    chatBody.appendChild(msg);
    chatBody.scrollTop = chatBody.scrollHeight;
  }

  // 💬 Show typing indicator
  function showTypingIndicator() {
    typingIndicator.classList.remove("hidden");
    chatBody.scrollTop = chatBody.scrollHeight;
  }

  // 📴 Hide typing indicator
  function hideTypingIndicator() {
    typingIndicator.classList.add("hidden");
  }

  // ✨ Format bot messages (convert markdown-like text to HTML)
  function formatBotMessage(text) {
    if (!text) return "";

    // Convert markdown-like lists and stars to clean HTML
    let formatted = text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>") // bold
      .replace(/\*(.*?)\*/g, "<em>$1</em>") // italics
      .replace(/(?:^|\n)- (.*?)(?=\n|$)/g, "<li>$1</li>") // list items
      .replace(/\n/g, "<br>"); // newlines

    // Wrap lists in <ul>
    if (formatted.includes("<li>")) {
      formatted = "<ul>" + formatted + "</ul>";
    }

    // Auto-detect and embed images if a property image URL is present
    const imageRegex = /(https?:\/\/[^\s]+\.(jpg|jpeg|png|webp))/gi;
    formatted = formatted.replace(imageRegex, (url) => {
      return `<img src="${url}" alt="Property image" class="property-image" loading="lazy">`;
    });

    return formatted;
  }
});
