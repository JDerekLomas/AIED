import './App.css'

const specLevelData = [
  { level: 'S1', name: 'Persona', errorRate: 4.9, targetRate: 0.0, description: '"You are a 6th grade student..."' },
  { level: 'S2', name: 'Knowledge State', errorRate: 5.0, targetRate: 0.0, description: 'Specifies what student knows/doesn\'t know' },
  { level: 'S3', name: 'Mental Model', errorRate: 43.8, targetRate: 88.6, description: 'Describes the misconception as a belief' },
  { level: 'S4', name: 'Production Rules', errorRate: 35.0, targetRate: 78.6, description: 'Step-by-step incorrect procedure' },
]

const humanBaseline = 38.2

function BarChart({ data, metric, label, baseline }: {
  data: typeof specLevelData,
  metric: 'errorRate' | 'targetRate',
  label: string,
  baseline?: number
}) {
  const max = Math.max(...data.map(d => d[metric]), baseline || 0)

  return (
    <div className="chart">
      <h3>{label}</h3>
      <div className="bars">
        {data.map(d => (
          <div key={d.level} className="bar-container">
            <div className="bar-label">{d.level}</div>
            <div className="bar-wrapper">
              <div
                className={`bar ${d[metric] > 70 ? 'high' : d[metric] > 30 ? 'medium' : 'low'}`}
                style={{ width: `${(d[metric] / max) * 100}%` }}
              >
                <span className="bar-value">{d[metric].toFixed(1)}%</span>
              </div>
            </div>
          </div>
        ))}
        {baseline && (
          <div className="bar-container baseline">
            <div className="bar-label">Human</div>
            <div className="bar-wrapper">
              <div
                className="bar human"
                style={{ width: `${(baseline / max) * 100}%` }}
              >
                <span className="bar-value">{baseline}%</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function App() {
  return (
    <div className="container">
      <header>
        <h1>Wrong for the Right Reasons</h1>
        <p className="subtitle">Can LLM Synthetic Students Exhibit Human-Like Misconceptions?</p>
        <p className="meta">AIED 2026 Pilot Results | Frontier Models (GPT-4o, Claude Sonnet 4)</p>
      </header>

      <section className="key-finding">
        <h2>Key Finding</h2>
        <div className="finding-box">
          <div className="finding-stat">88.6%</div>
          <div className="finding-text">
            <strong>S3 (Mental Model) prompts</strong> achieve the highest target misconception alignment when LLMs make errors
          </div>
        </div>
      </section>

      <section className="methodology">
        <h2>Specification Levels</h2>
        <div className="spec-grid">
          {specLevelData.map(d => (
            <div key={d.level} className={`spec-card ${d.level.toLowerCase()}`}>
              <div className="spec-header">
                <span className="spec-level">{d.level}</span>
                <span className="spec-name">{d.name}</span>
              </div>
              <p className="spec-desc">{d.description}</p>
              <div className="spec-stats">
                <div className="stat">
                  <span className="stat-value">{d.errorRate.toFixed(1)}%</span>
                  <span className="stat-label">Error Rate</span>
                </div>
                <div className="stat">
                  <span className="stat-value">{d.targetRate.toFixed(1)}%</span>
                  <span className="stat-label">Target Rate</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="results">
        <h2>Results</h2>
        <div className="charts">
          <BarChart data={specLevelData} metric="errorRate" label="Error Rate (% wrong answers)" />
          <BarChart data={specLevelData} metric="targetRate" label="Target Rate (% hitting target distractor, among errors)" baseline={humanBaseline} />
        </div>
      </section>

      <section className="interpretation">
        <h2>Interpretation</h2>
        <div className="insight-grid">
          <div className="insight">
            <h3>The Competence Paradox</h3>
            <p>Frontier models are too capable at S1/S2 - they rarely make errors. But when prompted with S3/S4 misconception specifications, they make errors that align with target misconceptions.</p>
          </div>
          <div className="insight">
            <h3>S3 vs S4 Tradeoff</h3>
            <p><strong>S3</strong> (Mental Model) produces highest target alignment (88.6%) but overshoots human error rates. <strong>S4</strong> (Production Rules) is closer to human baseline but slightly lower target alignment.</p>
          </div>
          <div className="insight">
            <h3>Binary Behavior</h3>
            <p>Frontier models show binary behavior: correct at S1/S2 (0% target hits because few errors), but when induced to err via S3/S4, they consistently hit the target misconception.</p>
          </div>
        </div>
      </section>

      <section className="data-table">
        <h2>Full Results Table</h2>
        <table>
          <thead>
            <tr>
              <th>Spec Level</th>
              <th>Description</th>
              <th>Error Rate</th>
              <th>Target Rate</th>
              <th>vs Human (38.2%)</th>
            </tr>
          </thead>
          <tbody>
            {specLevelData.map(d => (
              <tr key={d.level}>
                <td><strong>{d.level}</strong> {d.name}</td>
                <td>{d.description}</td>
                <td>{d.errorRate.toFixed(1)}%</td>
                <td className={d.targetRate > 70 ? 'highlight' : ''}>{d.targetRate.toFixed(1)}%</td>
                <td className={d.targetRate > humanBaseline ? 'positive' : 'negative'}>
                  {d.targetRate > humanBaseline ? '+' : ''}{(d.targetRate - humanBaseline).toFixed(1)} pp
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="table-note">N = 322 responses (82 per spec level x 2 models x 2 reps). Human baseline from Eedi diagnostic data.</p>
      </section>

      <footer>
        <p>Pilot experiment: January 26, 2026 | Models: GPT-4o, Claude Sonnet 4 | Items: 113 Eedi diagnostic questions</p>
        <p>Misconceptions: Order of Operations (1507), Negative Multiplication (1597), Fraction Addition (217)</p>
      </footer>
    </div>
  )
}

export default App
