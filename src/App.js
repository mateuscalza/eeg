import React, { useCallback, useState, useRef, useEffect } from 'react'
import * as tf from '@tensorflow/tfjs'
import { useAsync, useMeasure } from 'react-use'
import mapRange from './utils/mapRange'

const size = 512
const fileLinesOffset = 5
const fileLinesSkip = 768
const labels = ['Espícula', 'Normal', 'Piscada', 'Ruído']
const canvasPadding = 40

export default function App() {
  const modelResult = useAsync(() => tf.loadLayersModel('/model.json'), [])
  const [values, setValues] = useState(null)
  const [wrapperRef, { width, height }] = useMeasure()
  const canvasRef = useRef(null)
  const lastPointRef = useRef({ x: null, y: null })
  const model = modelResult.value

  const handleFileChange = useCallback(
    async (event) => {
      const file = event.target.files[0]
      if (!model || !file) {
        setValues(null)
        return
      }

      const fileContent = await file.text()
      const lines = fileContent
        .split(/\r?\n/)
        .filter(
          (value, index) =>
            index >= fileLinesOffset + fileLinesSkip &&
            index < fileLinesOffset + fileLinesSkip + size
        )
        .map(parseFloat)

      const min = Math.min(...lines)
      const max = Math.max(...lines)

      const normalized = lines.map((value) => (value - min) / (max - min))
      setValues(normalized)
    },
    [model]
  )

  const predictionResult = useAsync(async () => {
    if (!values || !model) {
      return null
    }

    const tensor = tf.tensor([values])
    const predictedTensor = await model.predict(tensor)
    const outputs = await predictedTensor.array()
    return outputs[0]
      .map((classPrediction, classIndex) => ({
        key: String(classIndex),
        prediction: classPrediction,
        formattedPercentage: `${(classPrediction * 100)
          .toFixed(3)
          .replace('.', ',')}%`,
        label: labels[classIndex],
      }))
      .sort((a, b) => b.prediction - a.prediction)
  }, [model, values])

  const canvasWidth = width - canvasPadding
  const canvasHeight = Math.min(300, height - canvasPadding)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvasWidth || !canvasHeight || !canvas) {
      return
    }

    const context = canvas.getContext('2d')
    // console.log(0, 0, canvasWidth, canvasHeight)
    context.clearRect(0, 0, canvasWidth, canvasHeight)

    if (!values) {
      return
    }
    const xStep = canvasWidth / size

    context.beginPath()
    for (let x = 0; x < values.length; x++) {
      const y = values[x]
      context[x === 0 ? 'moveTo' : 'lineTo'](x * xStep, canvasHeight * (1 - y))
    }
    context.lineWidth = 2
    context.strokeStyle = 'rgba(52,152,219,1.0)'
    context.stroke()
  }, [canvasRef, canvasWidth, canvasHeight, values])

  // console.log('values', values)

  const handleDraw = useCallback(
    (event) => {
      console.log(event.nativeEvent)
      const xStep = canvasWidth / size
      const y = mapRange(event.nativeEvent.offsetY, canvasHeight, 0, 0, 1)
      const x = parseInt(
        mapRange(event.nativeEvent.offsetX, 0, canvasWidth, 0, size),
        10
      )

      if (x < 0 || x > size - 1) {
        return
      }

      const lastPoint = lastPointRef.current
      lastPointRef.current = { x, y }
      setValues((oldValues) => {
        const newValues = oldValues
          ? Array.from(oldValues)
          : Array.from(Array(size)).fill(0.5)

        if (lastPoint.x !== null && lastPoint.y !== null) {
          const minX = Math.min(lastPoint.x, x)
          const maxX = Math.max(lastPoint.x, x)
          for (let subX = minX; subX <= maxX; subX++) {
            newValues[subX] = y
          }
        } else {
          newValues[x] = y
        }

        return newValues
      })
    },
    [canvasWidth, canvasHeight, lastPointRef.current]
  )
  const handleMouseMove = useCallback(
    (event) => {
      if (event.buttons !== 1) {
        return
      }
      handleDraw(event)
    },
    [handleDraw]
  )
  const handleStopMoving = useCallback(() => {
    lastPointRef.current = { x: null, y: null }
  }, [])

  useEffect(() => {
    if (!canvasRef.current) {
      return
    }

    const handleTouch = (event) => {
      event.preventDefault()
      const rect = event.target.getBoundingClientRect()
      const offsetX = event.targetTouches[0].pageX - rect.left
      const offsetY = event.targetTouches[0].pageY - rect.top
      handleDraw({ nativeEvent: { offsetX, offsetY } })
    }
    const handleTouchEnd = (event) => {
      event.preventDefault()
      handleStopMoving({ nativeEvent: event })
    }

    canvasRef.current.addEventListener('touchstart', handleTouch, false)
    canvasRef.current.addEventListener('touchmove', handleTouch, false)
    canvasRef.current.addEventListener('touchend', handleTouchEnd, false)

    return () => {
      if (!canvasRef.current) {
        return
      }
      canvasRef.current.removeEventListener('touchstart', handleTouch)
      canvasRef.current.removeEventListener('touchmove', handleTouch)
      canvasRef.current.removeEventListener('touchend', handleTouchEnd)
    }
  }, [canvasRef.current])

  if (modelResult.loading) {
    return <div className='app is-loading'>Carregando modelo...</div>
  }

  if (modelResult.error || predictionResult.error) {
    return (
      <div className='app is-error'>
        Ocorreu um erro:
        <br />
        {modelResult.error?.message || predictionResult.error?.message}
        <br />
        <button onClick={() => window.location.reload()}>Recarregar</button>
      </div>
    )
  }

  return (
    <div className='app'>
      <main ref={wrapperRef}>
        <canvas
          ref={canvasRef}
          width={canvasWidth}
          height={canvasHeight}
          onMouseDown={handleDraw}
          onMouseMove={handleMouseMove}
          onMouseUp={handleStopMoving}
          onMouseLeave={handleStopMoving}
          style={{
            left: '50%',
            marginLeft: -canvasWidth / 2,
            top: '50%',
            marginTop: -canvasHeight / 2,
          }}
        />
      </main>
      <aside>
        <input type='file' onChange={handleFileChange} />
        <ul>
          {(predictionResult.value || []).map(
            ({ key, label, formattedPercentage }) => (
              <li key={key}>
                {label}: {formattedPercentage}
              </li>
            )
          )}
        </ul>
      </aside>
    </div>
  )
}
