// ═══════════════════════════════════════════════════════════════════════════
// Tinah — Qarba AI Assistant  |  script.js
// ═══════════════════════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {

  // ── Element references ───────────────────────────────────────────────────
  const chatWidget    = document.getElementById("chat-widget");
  const chatToggleBtn = document.getElementById("chat-toggle-btn");
  const chatCloseBtn  = document.getElementById("chat-close-btn");
  const sendBtn       = document.getElementById("send-btn");
  const userInput     = document.getElementById("user-input");
  const chatBody      = document.getElementById("chat-body");

  // ── State ────────────────────────────────────────────────────────────────
  let isWaiting  = false;   // prevents double-sends while Tinah is responding
  let isOpen     = false;   // tracks widget open/closed state
  let wasDragged = false;   // distinguishes a drag from a click on toggle btn

  // Expose globally — belt-and-suspenders for any inline onclick in HTML
  window.sendMessage = sendMessage;

  // ── Toggle widget open/close ─────────────────────────────────────────────
  function openWidget() {
    isOpen = true;
    chatWidget.classList.add("is-open");
    // Re-trigger entrance animation every time widget opens
    chatWidget.style.animation = "none";
    requestAnimationFrame(() => {
      chatWidget.style.animation = "";
    });
    setTimeout(() => userInput.focus(), 120);
  }

  function closeWidget() {
    isOpen = false;
    chatWidget.classList.remove("is-open");
  }

  chatToggleBtn.addEventListener("click", () => {
    if (wasDragged) { wasDragged = false; return; }
    isOpen ? closeWidget() : openWidget();
  });

  chatCloseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    closeWidget();
  });

  // ── Send message ─────────────────────────────────────────────────────────
  async function sendMessage() {
    const message = userInput.value.trim();
    if (!message || isWaiting) return;

    // Input length guard
    if (message.length > 500) {
      appendMessage("Please keep your message under 500 characters.", "bot-message");
      return;
    }

    isWaiting        = true;
    sendBtn.disabled = true;
    appendMessage(message, "user-message");
    userInput.value  = "";
    userInput.style.height = "";
    userInput.style.overflowY = "hidden";

    // Create typing indicator fresh every time
    const typingEl = createTypingIndicator();
    chatBody.appendChild(typingEl);
    chatBody.scrollTop = chatBody.scrollHeight;

    try {
      const response = await fetch("/chat", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ message }),
      });

      const data = await response.json();
      typingEl.remove();
      appendMessage(formatBotMessage(data.response), "bot-message");

    } catch (err) {
      typingEl.remove();
      appendMessage(
        "I couldn't connect to the server. Please check your connection and try again.",
        "bot-message"
      );
      console.error("Chat error:", err);

    } finally {
      isWaiting        = false;
      sendBtn.disabled = false;
      userInput.focus();
    }
  }

  // Auto-resize textarea as user types
  userInput.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = this.scrollHeight + "px";
    this.style.overflowY = this.scrollHeight > 120 ? "auto" : "hidden";
  });

  // Event listeners — JS only, no duplicate inline HTML handlers
  sendBtn.addEventListener("click", sendMessage);
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // ── Append a message bubble ──────────────────────────────────────────────
  function appendMessage(content, className) {
    const msg     = document.createElement("div");
    msg.className = `message ${className}`;
    msg.innerHTML = content;
    chatBody.appendChild(msg);
    chatBody.scrollTop = chatBody.scrollHeight;

    msg.style.opacity   = "0";
    msg.style.transform = "translateY(10px)";
    requestAnimationFrame(() => {
      msg.style.transition = "opacity 0.28s ease, transform 0.28s ease";
      msg.style.opacity    = "1";
      msg.style.transform  = "translateY(0)";
    });
  }

  // ── Typing indicator ─────────────────────────────────────────────────────
  function createTypingIndicator() {
    const wrapper     = document.createElement("div");
    wrapper.className = "message bot-message typing-indicator-wrapper";
    wrapper.innerHTML = `
      <span class="typing-label">Tinah is typing</span>
      <span class="typing-dots">
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
      </span>
    `;
    return wrapper;
  }

  // ── Format bot messages ──────────────────────────────────────────────────
  function formatBotMessage(text) {
    if (!text) return "I didn't receive a response. Please try again.";

    // 1. Extract [IMAGE]URL[/IMAGE] tags — store as placeholders
    const images  = [];
    let formatted = text.replace(/\[IMAGE\](.*?)\[\/IMAGE\]/gs, (_, url) => {
      const idx = images.length;
      images.push(
        `<div class="property-image-wrapper">
           <img src="${url.trim()}"
                alt="Property image"
                class="property-image"
                loading="lazy"
                onerror="this.closest('.property-image-wrapper').style.display='none'">
         </div>`
      );
      return `%%IMAGE_${idx}%%`;
    });

    // 2. Markdown bold and italic
    formatted = formatted
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g,     "<em>$1</em>");

    // 3. Property card detection and rendering
    const blocks = formatted.split(/\n\n+/);
    const parts  = blocks.map(block => {
      const trimmed = block.trim();
      if (!trimmed) return "";

      if (trimmed.includes("Property:") && trimmed.includes("Price:")) {
        const lines = trimmed.split("\n").map(l => l.trim()).filter(Boolean);
        const rows  = lines.map(line => {
          if (line.startsWith("%%IMAGE_")) return line;
          const colonIdx = line.indexOf(":");
          if (colonIdx > 0 && colonIdx < 20) {
            const label = line.slice(0, colonIdx).trim();
            const value = line.slice(colonIdx + 1).trim();
            return `<div class="prop-row"><span class="prop-label">${label}</span><span class="prop-value">${value}</span></div>`;
          }
          return `<div class="prop-row">${line}</div>`;
        }).join("");
        return `<div class="property-card">${rows}</div>`;
      }

      return trimmed.replace(/\n/g, "<br>");
    });

    formatted = parts.filter(Boolean).join("<br>");

    // 4. Restore image HTML
    images.forEach((imgHtml, idx) => {
      formatted = formatted.replace(`%%IMAGE_${idx}%%`, imgHtml);
    });

    // 5. Linkify raw URLs not already in href/src
    formatted = formatted.replace(
      /(?<!href="|src=")(https?:\/\/[^\s<"]+)/g,
      '<a href="$1" target="_blank" rel="noopener noreferrer" class="chat-link">$1</a>'
    );

    return formatted;
  }


  // ═══════════════════════════════════════════════════════════════════════
  // DRAGGABLE — Chat Widget + Toggle Button
  // Uses Pointer Events API for unified mouse + touch support.
  //
  // HOW IT WORKS:
  // - pointerdown: record start position, switch element to left/top coords
  // - pointermove: compute delta, clamp within viewport, update position
  // - pointerup:   clean up, detect if it was a drag or just a click
  //
  // The toggle button sets wasDragged=true during a drag so its click
  // handler knows not to open/close the widget after dragging.
  // ═══════════════════════════════════════════════════════════════════════

  function makeDraggable(element, handle) {
    let startX, startY, startLeft, startTop;
    let isDragging = false;
    let dragMoved  = false;

    handle.addEventListener("pointerdown", onDragStart, { passive: false });

    function onDragStart(e) {
      if (e.button !== undefined && e.button !== 0) return;
      if (e.target.closest(".chat-close-btn")) return;

      isDragging = true;
      dragMoved  = false;

      const rect = element.getBoundingClientRect();
      startX     = e.clientX;
      startY     = e.clientY;

      // Normalise to left/top — element may currently use right/bottom CSS
      element.style.left   = rect.left + "px";
      element.style.top    = rect.top  + "px";
      element.style.right  = "auto";
      element.style.bottom = "auto";

      startLeft = rect.left;
      startTop  = rect.top;

      element.classList.add("is-dragging");
      handle.style.cursor = "grabbing";

      handle.setPointerCapture(e.pointerId);
      e.preventDefault();

      document.addEventListener("pointermove", onDragMove, { passive: false });
      document.addEventListener("pointerup",   onDragEnd);
    }

    function onDragMove(e) {
      if (!isDragging) return;

      const dx = e.clientX - startX;
      const dy = e.clientY - startY;

      if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
        dragMoved = true;
        if (element === chatToggleBtn) wasDragged = true;
      }

      const rect    = element.getBoundingClientRect();
      const maxLeft = window.innerWidth  - rect.width;
      const maxTop  = window.innerHeight - rect.height;

      const newLeft = Math.max(0, Math.min(maxLeft, startLeft + dx));
      const newTop  = Math.max(0, Math.min(maxTop,  startTop  + dy));

      element.style.left = newLeft + "px";
      element.style.top  = newTop  + "px";

      e.preventDefault();
    }

    function onDragEnd() {
      isDragging          = false;
      element.classList.remove("is-dragging");
      handle.style.cursor = "grab";

      document.removeEventListener("pointermove", onDragMove);
      document.removeEventListener("pointerup",   onDragEnd);

      if (!dragMoved && element === chatToggleBtn) {
        wasDragged = false;
      }
    }
  }

  // Attach drag to chat widget (via header handle) and toggle button
  const dragHandle = document.getElementById("chat-drag-handle");
  makeDraggable(chatWidget, dragHandle);
  makeDraggable(chatToggleBtn, chatToggleBtn);

});
