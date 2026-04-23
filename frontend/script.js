const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const result = document.getElementById("result");
const classifyBtn = document.getElementById("classifyBtn");
const clearBtn = document.getElementById("clearBtn");
const uploadText = document.getElementById("uploadText");

function setResult(message, status = "") {
    result.textContent = message;
    result.className = status;
}

function handleSelectedFile(file) {
    if (!file) {
        return;
    }
    const reader = new FileReader();
    reader.onload = function (event) {
        preview.src = event.target.result;
        preview.style.display = "block";
        classifyBtn.disabled = false;
        uploadText.textContent = file.name;
        setResult("");
    };
    reader.readAsDataURL(file);
}

imageInput.addEventListener("change", function () {
    handleSelectedFile(this.files[0]);
});

classifyBtn.addEventListener("click", async function () {
    const file = imageInput.files[0];
    if (!file) {
        setResult("Please select an image first.", "error");
        return;
    }

    classifyBtn.disabled = true;
    classifyBtn.textContent = "Classifying...";
    setResult("Sending image for prediction...");

    const formData = new FormData();
    formData.append("image", file);

    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }

        const data = await response.json();
        setResult(`Prediction: ${data.result}`, "success");
    } catch (error) {
        console.error("Prediction failed:", error);
        setResult(
            "Could not classify the image. Make sure the Flask API is running on port 5000.",
            "error"
        );
    } finally {
        classifyBtn.disabled = !imageInput.files[0];
        classifyBtn.textContent = "Classify Image";
    }
});

clearBtn.addEventListener("click", function () {
    imageInput.value = "";
    preview.src = "";
    preview.style.display = "none";
    classifyBtn.disabled = true;
    uploadText.textContent = "Click to choose an image";
    setResult("");
});
