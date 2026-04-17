imageInput = document.getElementById('imageInput');
preview = document.getElementById('preview');
result = document.getElementById('result');
button = document.getElementById('classifyBtn');

imageInput.addEventListener('change', function() {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
        }
        reader.readAsDataURL(file);
    }
});

preview.addEventListener('click', function() {
    imageInput.click();
});

button.addEventListener('click', function() {
    const file = imageInput.files[0];
    if (!file) {
        alert("Please select an image first.");
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        result.textContent = `Prediction: ${data.result}`;
    })
    .catch(error => {
        console.error('Error:', error);
        result.textContent = 'Error occurred during classification.';
    });
});
