import React, { useEffect, useRef } from 'react';

function HexagonalGrid() {
  const svgRef = useRef(null);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;

    const svgNS = "http://www.w3.org/2000/svg";
    const hexSize = 40;
    const hexWidth = hexSize * 2;
    const hexHeight = Math.sqrt(3) * hexSize;
    const hexHSpacing = hexWidth * 0.75;
    const hexVSpacing = hexHeight;

    function createHexagon(cx, cy) {
        const points = [];
        for (let i = 0; i < 6; i++) {
            const angle = (Math.PI / 3) * i;
            const x = cx + hexSize * Math.cos(angle);
            const y = cy + hexSize * Math.sin(angle);
            points.push(`${x},${y}`);
        }
        return points.join(' ');
    }

    function drawGrid() {
        if (!svg.parentElement) return;
        svg.innerHTML = '';
        const { width, height } = svg.parentElement.getBoundingClientRect();

        const cols = Math.ceil(width / hexHSpacing) + 3;
        const rows = Math.ceil(height / hexVSpacing) + 3;

        for (let row = -1; row < rows; row++) {
            for (let col = -1; col < cols; col++) {
                const x = col * hexHSpacing + hexSize;
                const y = row * hexVSpacing + hexSize + (col % 2) * (hexVSpacing / 2);
                
                // --- CREATE TWO POLYGONS ---
                const points = createHexagon(x, y);

                // 1. The Visible Display Hexagon
                const polygonDisplay = document.createElementNS(svgNS, 'polygon');
                polygonDisplay.setAttribute('points', points);
                polygonDisplay.setAttribute('class', 'hexagon-display');
                // Give it a unique ID so the sensor can find it
                const displayId = `hex-display-${row}-${col}`;
                polygonDisplay.setAttribute('id', displayId);

                // 2. The Invisible Sensor Hexagon
                const polygonSensor = document.createElementNS(svgNS, 'polygon');
                polygonSensor.setAttribute('points', points);
                polygonSensor.setAttribute('class', 'hexagon-sensor');

                // --- ADD EVENT LISTENERS TO THE SENSOR ---
                polygonSensor.addEventListener('mouseenter', function() {
                    // Find the corresponding display hexagon
                    const displayHex = document.getElementById(displayId);
                    if (!displayHex) return;

                    // Calculate movement
                    const svgCenterX = width / 2;
                    const svgCenterY = height / 2;
                    const dx = x - svgCenterX;
                    const dy = y - svgCenterY;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    const moveDistance = 15;
                    const moveX = (dx / (distance || 1)) * moveDistance;
                    const moveY = (dy / (distance || 1)) * moveDistance;
                    
                    // Animate the display hexagon
                    displayHex.style.transform = `translate(${moveX}px, ${moveY}px) scale(1.1)`;
                    displayHex.classList.add('is-hovered');
                });
                
                polygonSensor.addEventListener('mouseleave', function() {
                    // Find the corresponding display hexagon
                    const displayHex = document.getElementById(displayId);
                    if (!displayHex) return;

                    // Reset the display hexagon
                    displayHex.style.transform = 'translate(0, 0) scale(1)';
                    displayHex.classList.remove('is-hovered');
                });
                
                // Add both to the SVG. Display first, so Sensor is "on top" to catch clicks.
                svg.appendChild(polygonDisplay);
                svg.appendChild(polygonSensor);
            }
        }
    }

    drawGrid();
    const debouncedDrawGrid = () => setTimeout(drawGrid, 100);
    window.addEventListener('resize', debouncedDrawGrid);

    return () => {
        window.removeEventListener('resize', debouncedDrawGrid);
    };
}, []); // The empty array [] means this effect runs only once after the component mounts.

  return (
    <div className="hexagon-container">
      <svg id="hexagonGrid" ref={svgRef}></svg>
    </div>
  );
}

export default HexagonalGrid;