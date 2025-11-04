import { useMemo } from 'react';


const STAR_CONFIG = {
  totalStars: 15,
  
  // Base size distribution
  baseSizes: {
    tiny: { weight: 0.35, size: 1 },     // 35% start at 1px
    small: { weight: 0.40, size: 1.5 },  // 40% start at 1.5px
    medium: { weight: 0.25, size: 2.5 }    // 25% start at 2px
  },
  
  // Glow target sizes
  glowTargetSizes: {
    tiny: 3,      // 1px → 3px when glowing
    small: 5,   // 1.5px → 3.5px when glowing
    medium: 6   // 2px → 4px when glowing
  },
  
  // EXPANDED exclusion zones - keep stars AWAY from all content
  exclusionZones: [
    // Title "MEV" - expanded buffer
    { xRange: [40, 60], yRange: [3, 17] },
    
    // Keywords badges - expanded buffer
    { xRange: [28, 72], yRange: [14, 24] },
    
    // "What can I help with?" tagline - expanded buffer
    { xRange: [36, 64], yRange: [30, 40] },
    
    // Search bar area - LARGE buffer (main interaction zone)
    { xRange: [22, 78], yRange: [37, 54] },
    
    // Example prompts - expanded buffer
    { xRange: [26, 74], yRange: [58, 70] }
  ],
  
  // Minimum distance between stars (prevents clustering)
  minStarDistance: 12, // 12% of screen space
  
  // Grid-based distribution for better coverage
  gridDivisions: {
    rows: 5,
    cols: 6
  }
};

function StarField() {
  const stars = useMemo(() => {
    const generateStars = () => {
      const starArray = [];
      const sizeKeys = Object.keys(STAR_CONFIG.baseSizes);
      
      // Create cumulative distribution for base sizes
      let sizeCumulative = 0;
      const sizeThresholds = sizeKeys.map(key => {
        sizeCumulative += STAR_CONFIG.baseSizes[key].weight;
        return { key, threshold: sizeCumulative };
      });
      
      // Check if position is in ANY exclusion zone
      const isInExclusionZone = (x, y) => {
        return STAR_CONFIG.exclusionZones.some(zone => {
          const [xMin, xMax] = zone.xRange;
          const [yMin, yMax] = zone.yRange;
          return x >= xMin && x <= xMax && y >= yMin && y <= yMax;
        });
      };
      
      // Check if new star is too close to existing stars
      const isTooCloseToOthers = (x, y, existingStars) => {
        return existingStars.some(star => {
          const distance = Math.sqrt(
            Math.pow(x - star.x, 2) + Math.pow(y - star.y, 2)
          );
          return distance < STAR_CONFIG.minStarDistance;
        });
      };
      
      // Create a grid system for better distribution
      const { rows, cols } = STAR_CONFIG.gridDivisions;
      const cellWidth = 100 / cols;
      const cellHeight = 100 / rows;
      
      // Create array of grid cells (shuffled for randomness)
      const gridCells = [];
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          gridCells.push({ row, col });
        }
      }
      
      // Shuffle grid cells for random selection
      for (let i = gridCells.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [gridCells[i], gridCells[j]] = [gridCells[j], gridCells[i]];
      }
      
      // Generate stars using grid-based approach
      let cellIndex = 0;
      let attempts = 0;
      const maxTotalAttempts = 200;
      
      while (starArray.length < STAR_CONFIG.totalStars && attempts < maxTotalAttempts) {
        attempts++;
        
        // Get next grid cell
        const cell = gridCells[cellIndex % gridCells.length];
        cellIndex++;
        
        // Random position within this grid cell
        const x = cell.col * cellWidth + Math.random() * cellWidth;
        const y = cell.row * cellHeight + Math.random() * cellHeight;
        
        // Check validity
        if (isInExclusionZone(x, y)) continue;
        if (isTooCloseToOthers(x, y, starArray)) continue;
        
        // Determine base size
        const sizeRand = Math.random();
        const sizeResult = sizeThresholds.find(t => sizeRand <= t.threshold);
        const sizeKey = sizeResult.key;
        const baseSize = STAR_CONFIG.baseSizes[sizeKey].size;
        const glowSize = STAR_CONFIG.glowTargetSizes[sizeKey];
        
        // Random animation parameters
        const animationDelay = Math.random() * 6;
        const animationDuration = 4.5 + Math.random() * 2.5; // 4.5-7 seconds
        
        starArray.push({
          x,
          y,
          baseSize,
          glowSize,
          animationDelay,
          animationDuration,
          id: `star-${starArray.length}`
        });
      }
      
      return starArray;
    };
    
    return generateStars();
  }, []);

  return (
    <div className="star-field-container">
      {stars.map(star => (
        <div
          key={star.id}
          className="star-plus"
          style={{
            left: `${star.x}%`,
            top: `${star.y}%`,
            '--base-size': `${star.baseSize}px`,
            '--glow-size': `${star.glowSize}px`,
            '--animation-delay': `${star.animationDelay}s`,
            '--animation-duration': `${star.animationDuration}s`
          }}
        >
          <div className="star-plus-horizontal"></div>
          <div className="star-plus-vertical"></div>
        </div>
      ))}
    </div>
  );
}

export default StarField;