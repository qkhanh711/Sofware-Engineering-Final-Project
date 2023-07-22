document.getElementById("image-form").addEventListener("submit", async function(event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);

    const res = {};

    for(const [key, value] of formData) {
        res[key] = value;
    }

    console.log('before fetch')
    console.log(res);
    console.log('before fetch')
    const response = await fetch("/generateImage", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(res),
    });
    console.log('fetch done');

    const data = await response.json();
    console.log('fetch done');
    console.log(data);

    if (data.generated_result) {
        const generatedResult = data.generated_result;

        const resultDiv = document.getElementById("result");
        resultDiv.innerHTML = "";

        if (generatedResult.image) {
            const imgElement = document.createElement("img");
            imgElement.src = `.${generatedResult.image}`;
            imgElement.alt = "Generated Image";
            imgElement.className = "result-image";
            resultDiv.appendChild(imgElement);
        } else if (generatedResult.cap) {
            const pElement = document.createElement("p");
            pElement.textContent = generatedResult.cap;
            pElement.className = "result-caption";
            resultDiv.appendChild(pElement);
        }
    } else {
        console.log("No generated result available");
    }

    // Set input image path
    const model = res.model;
    const inputImagePath = `./Input_images/${model}/input.png`;
    const inputImageElement = document.getElementById("input-image");
    inputImageElement.src = inputImagePath;
});