// src/components/ImageGrid.jsx - FINAL VERSION FOR IN-PLACE ZOOM

import React, { useState } from 'react';

function ImageGrid({ images }) {
  // State to track the URL of the currently selected/zoomed image
  const [selectedImage, setSelectedImage] = useState(null);

  if (!images || images.length === 0) {
    return <p>No images found for this query.</p>;
  }

  // A dynamic class for the main grid container to know when an image is selected
  const gridContainerClass = `image-grid-container ${selectedImage ? 'has-selection' : ''}`;

  return (
    // The main container that will act as the positioning context
    <div className={gridContainerClass}>
      
      {/* This overlay is only visible when an image is selected, used to close the zoom */}
      {selectedImage && <div className="close-overlay" onClick={() => setSelectedImage(null)} />}

      <div className={`image-grid image-count-${images.length}`}>
        {images.map((imageUrl, index) => {
          // Determine if the current image is the selected one
          const isSelected = selectedImage === imageUrl;
          
          return (
            <div
              key={imageUrl || index}
              // Add dynamic classes to each card
              className={`image-card ${isSelected ? 'is-selected' : ''}`}
              onClick={() => setSelectedImage(imageUrl)}
            >
              <img 
                src={imageUrl} 
                alt={`Search result image ${index + 1}`} 
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default ImageGrid;   