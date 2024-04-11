import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setSelectedFiles(Array.from(e.target.files));
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) {
      alert('Please select at least one image to upload.');
      return;
    }

    setLoading(true);
    const newPredictions = [];

    for (const file of selectedFiles) {
      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await fetch('http://localhost:5528/predict', {
          method: 'POST',
          body: formData,
        });
        const data = await response.json();
        newPredictions.push(data);
      } catch (error) {
        console.error('Error fetching prediction:', error);
      }
    }

    setPredictions(newPredictions);
    setLoading(false);
  };

  return (
    <div className="App">
      <h1>Chest X-ray Image Classifier</h1>
      <input type="file" multiple onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload</button>
      {loading && <div className="loading">Processing...</div>}
      {predictions.map((prediction, index) => (
        <div key={index}>
          <h2>Prediction Result {index + 1}:</h2>
          <p>Class: {prediction.class}</p>
          <p>Confidence Score: {prediction.accuracy.toFixed(2)}%</p>
          <img
            src={`data:image/jpeg;base64,${prediction.imageData}`}
            alt={`Uploaded ${index + 1}`}
            style={{ maxWidth: '300px', maxHeight: '300px' }}
          />
        </div>
      ))}
    </div>
  );
}

export default App;