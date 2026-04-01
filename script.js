/* * Copyright (c) 2026 Khush Patil. All Rights Reserved.
 * This code is for personal/educational viewing only. 
 * Unauthorized copying, modification, or distribution is strictly prohibited.
 */
document.addEventListener('DOMContentLoaded', () => {
    // --- Smooth Scrolling for Navigation ---
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // --- Canvas Setup & Drawing Logic ---
    const canvas = document.getElementById('drawCanvas');
    const ctx = canvas.getContext('2d');
    let isDrawing = false;

    function initCanvas() {
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.lineWidth = 8;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = '#000000'; 
    }
    
    initCanvas();

    function getCoordinates(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.clientX || (e.touches && e.touches[0].clientX);
        const clientY = e.clientY || (e.touches && e.touches[0].clientY);
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }

    function startDrawing(e) {
        isDrawing = true;
        draw(e);
    }

    function stopDrawing() {
        isDrawing = false;
        ctx.beginPath(); 
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault(); 
        const { x, y } = getCoordinates(e);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
    }

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseout', stopDrawing);
    canvas.addEventListener('touchstart', startDrawing, { passive: false });
    canvas.addEventListener('touchend', stopDrawing);
    canvas.addEventListener('touchmove', draw, { passive: false });

    // --- Controls: Clear and Upload ---
    const clearBtn = document.getElementById('clearBtn');
    const imageUpload = document.getElementById('imageUpload');

    clearBtn.addEventListener('click', () => {
        initCanvas();
        resetDashboard();
    });

    imageUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = function(event) {
            const img = new Image();
            img.onload = function() {
                initCanvas();
                // Draw uploaded image stretched to fit canvas
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    });

    // --- REAL Inference Engine Integration ---
    const predictBtn = document.getElementById('predictBtn');
    const btnText = document.getElementById('btnText');
    const btnSpinner = document.getElementById('btnSpinner');
    const apiStatus = document.getElementById('apiStatus');
    const inferenceTime = document.getElementById('inference-time');

    predictBtn.addEventListener('click', async () => {
        setLoadingState(true);
        // Get the value from the new 3-way toggle
        const mode = document.querySelector('input[name="datasetMode"]:checked').value;
        const base64Image = canvas.toDataURL("image/png");

        try {
            const startTime = performance.now();
            
            // Sending POST request to your FastAPI backend
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    image: base64Image,
                    dataset_mode: mode 
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const results = await response.json();
            const endTime = performance.now();
            const latency = Math.round(endTime - startTime);

            updateDashboard(results, latency);

        } catch (error) {
            console.error("API Error:", error);
            alert("Failed to connect to the backend API. Is your FastAPI server running on http://localhost:8000?");
            resetDashboard();
        } finally {
            setLoadingState(false);
        }
    });

    function setLoadingState(isLoading) {
        predictBtn.disabled = isLoading;
        if (isLoading) {
            btnText.textContent = 'Processing...';
            btnSpinner.classList.remove('hidden');
            apiStatus.textContent = 'Running Inference';
            apiStatus.className = 'badge status-running';
            apiStatus.style.backgroundColor = '#f39c12'; 
        } else {
            btnText.textContent = 'Run Parallel Inference';
            btnSpinner.classList.add('hidden');
            apiStatus.textContent = 'Inference Complete';
            apiStatus.className = 'badge status-complete';
            apiStatus.style.backgroundColor = '#27ae60'; 
        }
    }

    // Helper function to calculate Red-to-Green gradient
    function getColorForConfidence(confidence) {
        const safeConf = Math.min(Math.max(confidence, 0), 100);
        const hue = (safeConf / 100) * 120;
        return `hsl(${hue}, 85%, 50%)`; 
    }

    function updateDashboard(results, latency) {
        inferenceTime.textContent = `${latency} ms`;

        // Update Labels
        document.getElementById('cnn-label').textContent = results.cnn.label;
        document.getElementById('vit-label').textContent = results.vit.label;
        document.getElementById('hyb-label').textContent = results.hybrid.label;

        // Update Percentage Text
        document.getElementById('cnn-conf-text').textContent = `${results.cnn.confidence}%`;
        document.getElementById('vit-conf-text').textContent = `${results.vit.confidence}%`;
        document.getElementById('hyb-conf-text').textContent = `${results.hybrid.confidence}%`;

        // Reset: Remove glow from all cards first
        const prefixes = ['cnn', 'vit', 'hyb'];
        prefixes.forEach(prefix => {
            document.getElementById(`card-${prefix}`).classList.remove('highlight-card');
            document.getElementById(`${prefix}-label`).classList.remove('glow-text');
        });

        // Find the winner
        let winnerPrefix = 'cnn';
        let maxConf = results.cnn.confidence;

        if (results.vit.confidence > maxConf) {
            winnerPrefix = 'vit';
            maxConf = results.vit.confidence;
        }
        if (results.hybrid.confidence > maxConf) {
            winnerPrefix = 'hyb';
            maxConf = results.hybrid.confidence;
        }

        // Apply the glow classes ONLY to the winner
        document.getElementById(`card-${winnerPrefix}`).classList.add('highlight-card');
        document.getElementById(`${winnerPrefix}-label`).classList.add('glow-text');
        
        // Animate Width and Apply Color Gradient
        setTimeout(() => {
            const cnnBar = document.getElementById('cnn-conf');
            cnnBar.style.width = `${results.cnn.confidence}%`;
            cnnBar.style.backgroundColor = getColorForConfidence(results.cnn.confidence);

            const vitBar = document.getElementById('vit-conf');
            vitBar.style.width = `${results.vit.confidence}%`;
            vitBar.style.backgroundColor = getColorForConfidence(results.vit.confidence);

            const hybBar = document.getElementById('hyb-conf');
            hybBar.style.width = `${results.hybrid.confidence}%`;
            hybBar.style.backgroundColor = getColorForConfidence(results.hybrid.confidence);
        }, 50); 
    }

    function resetDashboard() {
        apiStatus.textContent = 'Awaiting Input';
        apiStatus.className = 'badge status-idle';
        apiStatus.style.backgroundColor = '#7f8c8d'; 
        inferenceTime.textContent = '-- ms';

        const prefixes = ['cnn', 'vit', 'hyb'];
        
        prefixes.forEach(prefix => {
            document.getElementById(`${prefix}-label`).textContent = '--';
            document.getElementById(`${prefix}-conf-text`).textContent = '0%';
            
            const bar = document.getElementById(`${prefix}-conf`);
            bar.style.width = '0%';
            bar.style.backgroundColor = 'transparent';

            document.getElementById(`card-${prefix}`).classList.remove('highlight-card');
            document.getElementById(`${prefix}-label`).classList.remove('glow-text');
        });
    }
});