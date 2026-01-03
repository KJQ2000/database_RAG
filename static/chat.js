const chatWindow = document.getElementById("chatWindow");
const input = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");

sendBtn.onclick = sendMessage;
input.addEventListener("keydown", e => {
    if (e.key === "Enter") sendMessage();
});

function sendMessage() {
    const text = input.value.trim();
    if (!text) return;

    appendMessage(text, "user");
    input.value = "";

    const thinking = appendMessage("Thinking...", "bot");

    fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text })
    })
    .then(res => res.json())
    .then(data => {
        thinking.remove();
        appendMessage(marked.parse(data.answer), "bot", true);
    })
    .catch(() => {
        thinking.remove();
        appendMessage("Sorry, something went wrong.", "bot");
    });
}

function appendMessage(text, sender, isHTML = false) {
    const div = document.createElement("div");
    div.className = `message ${sender}`;
    if (isHTML) div.innerHTML = text;
    else div.innerText = text;

    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return div;
}
