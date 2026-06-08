import { useState } from 'react';

function ImageGrid({ images }) {
  const [selectedImage, setSelectedImage] = useState(null);

  if (!images || images.length === 0) {
    return <p>No images found for this query.</p>;
  }

  const gridContainerClass = `image-grid-container ${selectedImage ? 'has-selection' : ''}`;

  return (
    <div className={gridContainerClass}>
      
      {selectedImage && <div className="close-overlay" onClick={() => setSelectedImage(null)} />}

      <div className={`image-grid image-count-${images.length}`}>
        {images.map((imageUrl, index) => {
          const isSelected = selectedImage === imageUrl;
          
          return (
            <div
              key={imageUrl || index}
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
