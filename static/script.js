document.getElementById("image-form").addEventListener("submit", async function(event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);

    const res = {}

    for(const [key,value] of formData) {
        res[key] = value
    }

    console.log(res);

    const response = await fetch("/generateImage", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(res),
    });

    const data = await response.json();

    if (data.success) {
        const resultResponse = await fetch("/generateImage");
        const resultData = await resultResponse.json();

        if (resultData.generated_result) {
            const generatedResult = resultData.generated_result;

            const resultDiv = document.getElementById("result");
            resultDiv.innerHTML = "";

            if (generatedResult.hasOwnProperty("image")) {
                const imgElement = document.createElement("img");
                imgElement.src = generatedResult.image;
                resultDiv.appendChild(imgElement);
            } else if (generatedResult.hasOwnProperty("cap")) {
                const pElement = document.createElement("p");
                pElement.textContent = generatedResult.cap;
                resultDiv.appendChild(pElement);
            }
        } else {
            console.log("No generated result available");
        }
    } else {
        console.log("Failed to generate images");
    }
});