// drawUtils.js
const SKELETON_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],     // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],     // Index
    [0, 9], [9, 10], [10, 11], [11, 12],// Middle
    [0, 13], [13, 14], [14, 15], [15, 16], // Ring
    [0, 17], [17, 18], [18, 19], [19, 20]  // Pinky
];

export const drawDetection = (ctx, detection, frozenBox, canvasDim) => {
    ctx.clearRect(0, 0, canvasDim.w, canvasDim.h);

    // If we have a frozen box (the 2-second static box), draw it
    if (frozenBox) {
        const [x, y, w, h] = frozenBox;
        ctx.strokeStyle = '#00FF00'; // Green
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
    }

    // If we have live detection data, draw the skeleton inside
    if (detection) {
        const { keypoints, score } = detection;

        // Draw Skeleton (Cyan) - Always live so it tracks the hand smoothly
        ctx.strokeStyle = '#00FFFF';
        ctx.lineWidth = 2;
        ctx.beginPath();
        SKELETON_CONNECTIONS.forEach(([i, j]) => {
            const p1 = keypoints[i];
            const p2 = keypoints[j];
            if(p1.score > 0.3 && p2.score > 0.3) {
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
            }
        });
        ctx.stroke();

        // Draw Joints (Red)
        ctx.fillStyle = '#FF0000';
        keypoints.forEach(p => {
            if(p.score > 0.3) {
                ctx.beginPath();
                ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
                ctx.fill();
            }
        });

        // Draw Label on top of the frozen box
        if (frozenBox) {
            ctx.fillStyle = '#00FF00';
            ctx.font = '16px Arial';
            ctx.fillText(`Conf: ${(score * 100).toFixed(0)}%`, frozenBox[0], frozenBox[1] - 10);
        }
    }
};