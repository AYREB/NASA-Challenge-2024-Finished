document.getElementById('csvFile').onchange = () => {
    const fileInput = document.getElementById('csvFile');
    const fileName = fileInput.files[0]?.name || ''; // Get the selected file name
    document.getElementById('fileName').textContent = fileName ? `Selected File: ${fileName}` : ''; // Display file name
};

document.getElementById('uploadButton').onclick = async () => {
    const fileInput = document.getElementById('csvFile');
    const file = fileInput.files[0];

    if (!file) {
        displayError('Please select a CSV file to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Show spinner and hide image and chart title initially
    document.getElementById('spinner').style.display = 'block';
    document.getElementById('chartTitle').style.display = 'none';
    document.getElementById('image').style.display = 'none';

    try {
        const response = await fetch('https://nasa.thebayre.com:5000/seismic_detection', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const blob = await response.blob();
        const imgUrl = URL.createObjectURL(blob);
        const imageElement = document.getElementById('image');
        imageElement.src = imgUrl;
        imageElement.style.display = 'block'; // Show the image

        clearError();
    } catch (error) {
        console.error('Error:', error);
        displayError('There was an error uploading the file. Please try again.');
    } finally {
        // Hide spinner
        document.getElementById('spinner').style.display = 'none';
    }
};

function displayError(message) {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
}

function clearError() {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = '';
}
