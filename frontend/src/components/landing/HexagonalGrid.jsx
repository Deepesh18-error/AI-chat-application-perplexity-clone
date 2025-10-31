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
                
                const points = createHexagon(x, y);

                // 1. The Visible Display Hexagon
                const polygonDisplay = document.createElementNS(svgNS, 'polygon');
                polygonDisplay.setAttribute('points', points);
                polygonDisplay.setAttribute('class', 'hexagon-display');
                
                // 2. The Invisible Sensor Hexagon for mouse events
                const polygonSensor = document.createElementNS(svgNS, 'polygon');
                polygonSensor.setAttribute('points', points);
                polygonSensor.setAttribute('class', 'hexagon-sensor');

                // --- EVENT LISTENERS ---

                // When the mouse enters the invisible sensor...
                polygonSensor.addEventListener('mouseenter', () => {
                    // ...affect the visible display hexagon.
                    // THE FIX: Only apply scale. No translate().
                    polygonDisplay.style.transform = 'scale(1.15)';
                    polygonDisplay.classList.add('is-hovered');
                });
                
                // When the mouse leaves the invisible sensor...
                polygonSensor.addEventListener('mouseleave', () => {
                    // ...reset the visible display hexagon.
                    polygonDisplay.style.transform = 'scale(1)';
                    polygonDisplay.classList.remove('is-hovered');
                });
                
                svg.appendChild(polygonDisplay);
                svg.appendChild(polygonSensor); // Sensor on top to catch events
            }
        }
    }

    // --- EFFECT ORCHESTRATION ---
    drawGrid(); // Initial draw

    let resizeTimer;
    const debouncedDrawGrid = () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(drawGrid, 100);
    };
    window.addEventListener('resize', debouncedDrawGrid);

    // Cleanup function
    return () => {
        window.removeEventListener('resize', debouncedDrawGrid);
    };
}, []);

  return (
    <div className="hexagon-container">
      <svg id="hexagonGrid" ref={svgRef}></svg>
    </div>
  );
}

export default HexagonalGrid;