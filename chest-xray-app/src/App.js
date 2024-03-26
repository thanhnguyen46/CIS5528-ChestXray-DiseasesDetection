import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [imageData, setImageData] = useState(null);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    const reader = new FileReader();
    reader.onload = () => {
      setImageData(reader.result); // Add this line
    };
    reader.readAsDataURL(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Please select an image to upload.');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      console.log('Response data:', data); // DEBUG ON CONSOLE
      setPrediction(data);
    } catch (error) {
      console.error('Error fetching prediction:', error);
    }
  };

  return (
    <div className="App">
      <h1>Chest X-ray Image Classifier</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload</button>
      {prediction && prediction.class && prediction.accuracy && (
        <div>
          <h2>Prediction Result:</h2>
          <p>Class: {prediction.class}</p>
          <p>Accuracy: {(prediction.accuracy).toFixed(2)}%</p>
          {imageData && ( // Add this condition
            <img
              src={imageData}
              alt="Uploaded"
              style={{ maxWidth: '300px', maxHeight: '300px' }}
            />
          )}
        </div>
      )}
    </div>
  );
}

export default App;