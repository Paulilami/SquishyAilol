document.getElementById('send-btn').addEventListener('click', () => {
    const userInput = document.getElementById('user-input').value;
    document.getElementById('chat-output').innerHTML += `<div>User: ${userInput}</div>`;
    fetch('/api/selector', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ user_input: userInput })
    })
    .then(response => response.json())
    .then(selectorResponse => {
        document.getElementById('chat-output').innerHTML += `<div>Selector: ${JSON.stringify(selectorResponse)}</div>`;
        return fetch('/api/target', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ target: selectorResponse.Target, prompt: selectorResponse.Prompt })
        });
    })
    .then(response => response.json())
    .then(targetResponse => {
        document.getElementById('chat-output').innerHTML += `<div>Response: ${JSON.stringify(targetResponse)}</div>`;
    })
    .catch(error => {
        document.getElementById('chat-output').innerHTML += `<div>Error: ${error}</div>`;
    });
});

document.getElementById('stop-btn').addEventListener('click', () => {
    document.getElementById('chat-output').innerHTML += `<div>Command: Stop</div>`;
    fetch('/api/target', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ prompt: 'quit' })
    });
});
