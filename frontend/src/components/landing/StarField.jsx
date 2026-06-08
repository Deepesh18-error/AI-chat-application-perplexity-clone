import { useMemo } from 'react';

const STAR_CONFIG = {
  totalStars: 15,
  baseSizes: {
    tiny: { weight: 0.35, size: 1 },
    small: { weight: 0.4, size: 1.5 },
    medium: { weight: 0.25, size: 2.5 },
  },
  glowTargetSizes: {
    tiny: 3,
    small: 5,
    medium: 6,
  },
  exclusionZones: [
    { xRange: [40, 60], yRange: [3, 17] },
    { xRange: [28, 72], yRange: [14, 24] },
    { xRange: [36, 64], yRange: [30, 40] },
    { xRange: [22, 78], yRange: [37, 54] },
    { xRange: [26, 74], yRange: [58, 70] },
  ],
  minStarDistance: 12,
  gridDivisions: {
    rows: 5,
    cols: 6,
  },
};

function StarField() {
  const stars = useMemo(() => {
    const starArray = [];
    const sizeKeys = Object.keys(STAR_CONFIG.baseSizes);
    let sizeCumulative = 0;
    const sizeThresholds = sizeKeys.map((key) => {
      sizeCumulative += STAR_CONFIG.baseSizes[key].weight;
      return { key, threshold: sizeCumulative };
    });

    const isInExclusionZone = (x, y) => (
      STAR_CONFIG.exclusionZones.some((zone) => {
        const [xMin, xMax] = zone.xRange;
        const [yMin, yMax] = zone.yRange;
        return x >= xMin && x <= xMax && y >= yMin && y <= yMax;
      })
    );

    const isTooCloseToOthers = (x, y) => (
      starArray.some((star) => {
        const distance = Math.sqrt(
          Math.pow(x - star.x, 2) + Math.pow(y - star.y, 2)
        );
        return distance < STAR_CONFIG.minStarDistance;
      })
    );

    const { rows, cols } = STAR_CONFIG.gridDivisions;
    const cellWidth = 100 / cols;
    const cellHeight = 100 / rows;
    const gridCells = [];

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        gridCells.push({ row, col });
      }
    }

    for (let i = gridCells.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [gridCells[i], gridCells[j]] = [gridCells[j], gridCells[i]];
    }

    let cellIndex = 0;
    let attempts = 0;
    const maxTotalAttempts = 200;

    while (starArray.length < STAR_CONFIG.totalStars && attempts < maxTotalAttempts) {
      attempts++;

      const cell = gridCells[cellIndex % gridCells.length];
      cellIndex++;

      const x = cell.col * cellWidth + Math.random() * cellWidth;
      const y = cell.row * cellHeight + Math.random() * cellHeight;

      if (isInExclusionZone(x, y)) continue;
      if (isTooCloseToOthers(x, y)) continue;

      const sizeRand = Math.random();
      const sizeResult = sizeThresholds.find((threshold) => sizeRand <= threshold.threshold);
      const sizeKey = sizeResult.key;
      const baseSize = STAR_CONFIG.baseSizes[sizeKey].size;
      const glowSize = STAR_CONFIG.glowTargetSizes[sizeKey];
      const animationDelay = Math.random() * 6;
      const animationDuration = 4.5 + Math.random() * 2.5;

      starArray.push({
        x,
        y,
        baseSize,
        glowSize,
        animationDelay,
        animationDuration,
        id: `star-${starArray.length}`,
      });
    }

    return starArray;
  }, []);

  return (
    <div className="star-field-container">
      {stars.map((star) => (
        <div
          key={star.id}
          className="star-plus"
          style={{
            left: `${star.x}%`,
            top: `${star.y}%`,
            '--base-size': `${star.baseSize}px`,
            '--glow-size': `${star.glowSize}px`,
            '--animation-delay': `${star.animationDelay}s`,
            '--animation-duration': `${star.animationDuration}s`,
          }}
        >
          <div className="star-plus-horizontal" />
          <div className="star-plus-vertical" />
        </div>
      ))}
    </div>
  );
}

export default StarField;
