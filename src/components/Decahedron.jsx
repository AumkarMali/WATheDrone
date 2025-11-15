import { useRef, useEffect, useMemo } from 'react'
import './Decahedron.css'

function Decahedron({ probabilities = {} }) {
  const canvasRef = useRef(null)
  const droneLetters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
  
  // Normalize probabilities - use mock data if all are zero (for demo)
  const normalizedProbabilities = useMemo(() => {
    const hasData = droneLetters.some(letter => 
      typeof probabilities[letter] === 'number' && probabilities[letter] > 0
    )
    
    // If no real data, generate random demo probabilities
    if (!hasData) {
      const mockProbs = {}
      let total = 0
      
      // Generate random probabilities
      droneLetters.forEach(letter => {
        const prob = Math.random() * 30
        mockProbs[letter] = Math.round(prob * 10) / 10
        total += mockProbs[letter]
      })
      
      // Make one drone the clear winner (60-90%)
      const topIndex = Math.floor(Math.random() * 10)
      const topLetter = droneLetters[topIndex]
      mockProbs[topLetter] = Math.round((60 + Math.random() * 30) * 10) / 10
      
      // Redistribute remaining
      const remaining = 100 - mockProbs[topLetter]
      const otherTotal = Object.values(mockProbs).reduce((sum, val, idx) => 
        idx === topIndex ? sum : sum + val, 0)
      
      droneLetters.forEach((letter, idx) => {
        if (idx !== topIndex) {
          mockProbs[letter] = Math.round((mockProbs[letter] / otherTotal * remaining) * 10) / 10
        }
      })
      
      // Ensure sum is 100
      let finalSum = Object.values(mockProbs).reduce((a, b) => a + b, 0)
      const diff = 100 - finalSum
      mockProbs[topLetter] = Math.round((mockProbs[topLetter] + diff) * 10) / 10
      
      return mockProbs
    }
    
    // Use real probabilities
    return droneLetters.reduce((acc, letter) => {
      acc[letter] = typeof probabilities[letter] === 'number' ? probabilities[letter] : 0
      return acc
    }, {})
  }, [probabilities])

  // Find the drone with highest probability
  const topPrediction = useMemo(() => {
    let maxProb = -1
    let topDrone = null
    droneLetters.forEach(letter => {
      const prob = normalizedProbabilities[letter] || 0
      if (prob > maxProb) {
        maxProb = prob
        topDrone = letter
      }
    })
    return { letter: topDrone, probability: maxProb }
  }, [normalizedProbabilities])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    // High resolution canvas (1980x1980)
    const dpr = window.devicePixelRatio || 1
    const displayWidth = 500
    const displayHeight = 500
    const actualWidth = 1980
    const actualHeight = 1980

    // Set canvas size
    canvas.width = actualWidth
    canvas.height = actualHeight
    canvas.style.width = `${displayWidth}px`
    canvas.style.height = `${displayHeight}px`

    const ctx = canvas.getContext('2d')
    
    // Clear canvas at actual resolution
    ctx.clearRect(0, 0, actualWidth, actualHeight)
    
    // Scale context for high DPI rendering
    ctx.scale(actualWidth / displayWidth, actualHeight / displayHeight)

    const width = displayWidth
    const height = displayHeight
    const centerX = width / 2
    const centerY = height / 2
    const radius = Math.min(width, height) * 0.30

    // Number of axes (one for each drone)
    const numAxes = 10
    const angleStep = (2 * Math.PI) / numAxes
    const startAngle = -Math.PI / 2 // Start from top

    // Draw background grid circles
    for (let i = 1; i <= 5; i++) {
      const circleRadius = (radius * i) / 5
      ctx.beginPath()
      ctx.arc(centerX, centerY, circleRadius, 0, 2 * Math.PI)
      ctx.strokeStyle = `rgba(148, 163, 184, ${0.15 - i * 0.02})`
      ctx.lineWidth = 1
      ctx.stroke()
    }

    // Draw axes (lines from center to edge)
    for (let i = 0; i < numAxes; i++) {
      const angle = startAngle + (i * angleStep)
      const x = centerX + radius * Math.cos(angle)
      const y = centerY + radius * Math.sin(angle)
      
      ctx.beginPath()
      ctx.moveTo(centerX, centerY)
      ctx.lineTo(x, y)
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)'
      ctx.lineWidth = 1
      ctx.stroke()
    }

    // Calculate points for radar polygon
    const points = []
    droneLetters.forEach((letter, index) => {
      const probability = normalizedProbabilities[letter] || 0
      const angle = startAngle + (index * angleStep)
      // Scale distance more aggressively - minimum 15% of radius, max 95% of radius
      const minDistance = radius * 0.15
      const maxDistance = radius * 0.95
      const distance = minDistance + (probability / 100) * (maxDistance - minDistance)
      const x = centerX + distance * Math.cos(angle)
      const y = centerY + distance * Math.sin(angle)
      points.push({ x, y, letter, probability, angle, distance })
    })

    // Draw filled radar polygon with gradient
    if (points.length > 0) {
      const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius)
      gradient.addColorStop(0, 'rgba(99, 102, 241, 0.4)')
      gradient.addColorStop(0.5, 'rgba(168, 85, 247, 0.3)')
      gradient.addColorStop(1, 'rgba(99, 102, 241, 0.1)')

      ctx.beginPath()
      ctx.moveTo(points[0].x, points[0].y)
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y)
      }
      ctx.closePath()
      ctx.fillStyle = gradient
      ctx.fill()

      // Draw radar outline
      ctx.beginPath()
      ctx.moveTo(points[0].x, points[0].y)
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y)
      }
      ctx.closePath()
      ctx.strokeStyle = 'rgba(99, 102, 241, 0.8)'
      ctx.lineWidth = 3
      ctx.stroke()

      // Draw connecting lines to center
      points.forEach(point => {
        if (point.distance > 0) {
          ctx.beginPath()
          ctx.moveTo(centerX, centerY)
          ctx.lineTo(point.x, point.y)
          ctx.strokeStyle = 'rgba(99, 102, 241, 0.4)'
          ctx.lineWidth = 1.5
          ctx.stroke()
        }
      })
    }

    // Draw points and labels
    points.forEach((point, index) => {
      const { x, y, letter, probability, angle } = point
      const isTopPrediction = topPrediction.letter === letter && probability > 0

      // Draw point
      if (probability > 0) {
        ctx.beginPath()
        ctx.arc(x, y, isTopPrediction ? 8 : 6, 0, 2 * Math.PI)
        
        if (isTopPrediction) {
          const pointGradient = ctx.createRadialGradient(x, y, 0, x, y, 8)
          pointGradient.addColorStop(0, 'rgba(99, 102, 241, 1)')
          pointGradient.addColorStop(1, 'rgba(168, 85, 247, 0.8)')
          ctx.fillStyle = pointGradient
          ctx.shadowBlur = 15
          ctx.shadowColor = 'rgba(99, 102, 241, 0.8)'
        } else {
          ctx.fillStyle = probability > 50 ? 'rgba(99, 102, 241, 0.9)' : 
                         probability > 25 ? 'rgba(168, 85, 247, 0.8)' : 
                         'rgba(148, 163, 184, 0.7)'
          ctx.shadowBlur = 0
        }
        
        ctx.fill()
        ctx.strokeStyle = '#ffffff'
        ctx.lineWidth = 2
        ctx.stroke()
        ctx.shadowBlur = 0
      }

      // Draw label at the end of axis - positioned to fit within canvas
      const labelRadius = radius * 1.18
      const labelX = centerX + labelRadius * Math.cos(angle)
      const labelY = centerY + labelRadius * Math.sin(angle)

      // Background for label - larger to accommodate text
      ctx.fillStyle = 'rgba(15, 23, 42, 0.9)'
      ctx.beginPath()
      ctx.arc(labelX, labelY, 35, 0, 2 * Math.PI)
      ctx.fill()
      
      if (isTopPrediction) {
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.8)'
        ctx.lineWidth = 2.5
        ctx.stroke()
      }

      // Drone letter - larger font for better visibility
      ctx.fillStyle = probability > 0 ? '#c7d2fe' : 'rgba(148, 163, 184, 0.6)'
      ctx.font = 'bold 22px sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(letter, labelX, labelY - 8)

      // Probability percentage
      ctx.fillStyle = probability > 0 ? '#818cf8' : 'rgba(148, 163, 184, 0.5)'
      ctx.font = 'bold 14px sans-serif'
      ctx.fillText(`${Math.round(probability)}%`, labelX, labelY + 10)

      // Top prediction star
      if (isTopPrediction) {
        ctx.fillStyle = 'rgba(99, 102, 241, 1)'
        ctx.beginPath()
        ctx.arc(labelX, labelY - 18, 6, 0, 2 * Math.PI)
        ctx.fill()
        ctx.fillStyle = '#ffffff'
        ctx.font = 'bold 8px sans-serif'
        ctx.fillText('★', labelX, labelY - 17)
      }
    })

    // Draw center circle
    ctx.beginPath()
    ctx.arc(centerX, centerY, 10, 0, 2 * Math.PI)
    const centerGradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, 10)
    centerGradient.addColorStop(0, 'rgba(99, 102, 241, 0.8)')
    centerGradient.addColorStop(1, 'rgba(168, 85, 247, 0.6)')
    ctx.fillStyle = centerGradient
    ctx.fill()
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = 2
    ctx.stroke()

  }, [normalizedProbabilities, topPrediction])

  return (
    <div className="decahedron-container">
      <div className="decahedron-header">
        <h3 className="decahedron-title">Drone Probability</h3>
        <span className="decahedron-subtitle">Radar Chart Analysis</span>
      </div>

      {/* Top Prediction Display */}
      {topPrediction.letter && topPrediction.probability > 0 && (
        <div className="top-prediction-banner">
          <div className="prediction-label">Top Prediction</div>
          <div className="prediction-value">
            <span className="drone-letter-large">{topPrediction.letter}</span>
            <span className="probability-large">{Math.round(topPrediction.probability)}%</span>
          </div>
        </div>
      )}

      {/* Canvas for Radar Chart */}
      <div className="canvas-wrapper">
        <canvas
          ref={canvasRef}
          className="decahedron-canvas"
        />
      </div>

      <div className="decahedron-legend">
        <div className="legend-note">
          Each axis represents a drone (A-J). Distance from center indicates probability confidence.
        </div>
        <div className="legend-colors">
          <div className="legend-item">
            <span className="legend-color" style={{ background: 'rgba(148, 163, 184, 0.3)' }}></span>
            <span>0%</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ background: 'rgba(168, 85, 247, 0.6)' }}></span>
            <span>1-25%</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ background: 'rgba(168, 85, 247, 0.8)' }}></span>
            <span>25-50%</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ background: 'rgba(99, 102, 241, 0.9)' }}></span>
            <span>&gt;50%</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Decahedron
