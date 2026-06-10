import React, { useState, useEffect, useCallback, useRef } from 'react';
import * as d3 from 'd3';

// ─── API ─────────────────────────────────────────────────────────────────────

const API_BASE = process.env.REACT_APP_API_URL || '';
// const API_BASE = "https://dl-final-project-09xp.onrender.com";

async function fetchStockData(stockNo, months = 1) {
  const res = await fetch(`${API_BASE}/api/stock/${stockNo}?months=${months}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ─── Model config ─────────────────────────────────────────────────────────────

const MODEL_CONFIG = {
  LSTM: { color: '#7C3AED', label: 'LSTM', dash: '0' },
  GRU: { color: '#EA580C', label: 'GRU', dash: '0' },
  BiLSTM: { color: '#0891B2', label: 'BiLSTM', dash: '0' },
  Transformer: { color: '#059669', label: 'Transformer', dash: '0' },
};

// ─── K 線圖 + 預測線（D3）────────────────────────────────────────────────────

function CandlestickChart({ data, predictions, modelVisibility }) {
  const svgRef = useRef(null);
  const wrapRef = useRef(null);

  useEffect(() => {
    if (!data.length || !svgRef.current || !wrapRef.current) return;

    const margin = { top: 20, right: 20, bottom: 60, left: 60 };
    const totalWidth = wrapRef.current.clientWidth || 800;
    const totalHeight = 380;
    const width = totalWidth - margin.left - margin.right;
    const height = totalHeight - margin.top - margin.bottom;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', totalWidth)
      .attr('height', totalHeight)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const parseDate = d3.timeParse('%Y-%m-%d');
    const parsed = data.map(d => ({ ...d, dateObj: parseDate(d.date) })).filter(d => d.dateObj);

    // ── X 軸只用 K 線本身的日期，預測線對齊到相同日期 ──────────────────────
    const candleDates = parsed.map(d => d.date);
    const candleDateSet = new Set(candleDates);

    const xScale = d3.scaleBand()
      .domain(candleDates)
      .range([0, width])
      .padding(0.2);

    // Y scale: 同時涵蓋 K 線與預測值
    let yMinVals = [d3.min(parsed, d => d.low) * 0.995];
    let yMaxVals = [d3.max(parsed, d => d.high) * 1.005];
    Object.entries(predictions).forEach(([key, arr]) => {
      if (modelVisibility[key] && arr.length) {
        const visible = arr.filter(p => candleDateSet.has(p.date));
        if (visible.length) {
          yMinVals.push(d3.min(visible, d => d.predicted_close) * 0.995);
          yMaxVals.push(d3.max(visible, d => d.predicted_close) * 1.005);
        }
      }
    });

    const yScale = d3.scaleLinear()
      .domain([Math.min(...yMinVals), Math.max(...yMaxVals)])
      .range([height, 0]);

    // Grid
    svg.append('g')
      .call(d3.axisLeft(yScale).tickSize(-width).tickFormat(''))
      .call(g => g.select('.domain').remove())
      .call(g => g.selectAll('line').attr('stroke', '#f0f0f0').attr('stroke-dasharray', '3,3'));

    const tickEvery = Math.ceil(candleDates.length / 12);
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale)
        .tickValues(candleDates.filter((_, i) => i % tickEvery === 0))
        .tickFormat(d => d.slice(5)))
      .call(g => g.select('.domain').attr('stroke', '#ddd'))
      .call(g => g.selectAll('text')
        .attr('transform', 'rotate(-40)')
        .style('text-anchor', 'end')
        .attr('dx', '-0.5em')
        .attr('dy', '0.15em')
        .attr('font-size', 11)
        .attr('fill', '#888'));

    svg.append('g')
      .call(d3.axisLeft(yScale).ticks(6).tickFormat(d => d.toFixed(0)))
      .call(g => g.select('.domain').attr('stroke', '#ddd'))
      .call(g => g.selectAll('text').attr('font-size', 11).attr('fill', '#888'));

    const bandWidth = xScale.bandwidth();
    const tooltip = d3.select('body').select('#candle-tooltip');

    // Wicks
    svg.selectAll('.wick')
      .data(parsed)
      .join('line')
      .attr('class', 'wick')
      .attr('x1', d => xScale(d.date) + bandWidth / 2)
      .attr('x2', d => xScale(d.date) + bandWidth / 2)
      .attr('y1', d => yScale(d.high))
      .attr('y2', d => yScale(d.low))
      .attr('stroke', d => d.close >= d.open ? '#C0392B' : '#27AE60')
      .attr('stroke-width', 1.5);

    // Candle bodies
    svg.selectAll('.candle')
      .data(parsed)
      .join('rect')
      .attr('class', 'candle')
      .attr('x', d => xScale(d.date))
      .attr('y', d => yScale(Math.max(d.open, d.close)))
      .attr('width', bandWidth)
      .attr('height', d => Math.max(Math.abs(yScale(d.open) - yScale(d.close)), 1))
      .attr('fill', d => d.close >= d.open ? '#C0392B' : '#27AE60')
      .attr('rx', 1)
      .on('mouseover', (event, d) => {
        tooltip
          .style('display', 'block')
          .html(`
            <div style="font-weight:600;margin-bottom:6px">${d.date}</div>
            <div>開盤：<b>${d.open?.toFixed(2)}</b></div>
            <div>最高：<b style="color:#C0392B">${d.high?.toFixed(2)}</b></div>
            <div>最低：<b style="color:#27AE60">${d.low?.toFixed(2)}</b></div>
            <div>收盤：<b>${d.close?.toFixed(2)}</b></div>
            <div>成交量：<b>${d.volume?.toLocaleString()} 張</b></div>
            <div>漲跌幅：<b style="color:${d.daily_change_pct >= 0 ? '#C0392B' : '#27AE60'}">${d.daily_change_pct >= 0 ? '+' : ''}${d.daily_change_pct?.toFixed(2)}%</b></div>
          `);
      })
      .on('mousemove', (event) => {
        tooltip.style('left', (event.pageX + 14) + 'px').style('top', (event.pageY - 28) + 'px');
      })
      .on('mouseout', () => tooltip.style('display', 'none'));

    // ── 預測線：只保留與 K 線日期相符的點 ──────────────────────────────────
    const lineGen = d3.line()
      .x(d => xScale(d.date) + bandWidth / 2)
      .y(d => yScale(d.predicted_close))
      .curve(d3.curveMonotoneX);

    Object.entries(MODEL_CONFIG).forEach(([key, cfg]) => {
      if (!modelVisibility[key]) return;
      const pts = (predictions[key] || []).filter(p => candleDateSet.has(p.date));
      if (!pts.length) return;

      svg.append('path')
        .datum(pts)
        .attr('fill', 'none')
        .attr('stroke', cfg.color)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', cfg.dash)
        .attr('d', lineGen)
        .attr('opacity', 0)
        .transition().duration(600)
        .attr('opacity', 1);

      // Hover dots
      svg.selectAll(`.pred-dot-${key}`)
        .data(pts)
        .join('circle')
        .attr('class', `pred-dot-${key}`)
        .attr('cx', d => xScale(d.date) + bandWidth / 2)
        .attr('cy', d => yScale(d.predicted_close))
        .attr('r', 3)
        .attr('fill', cfg.color)
        .attr('opacity', 0.85)
        .on('mouseover', (event, d) => {
          tooltip
            .style('display', 'block')
            .html(`
              <div style="font-weight:600;margin-bottom:6px;color:${cfg.color}">${cfg.label} 預測</div>
              <div>日期：<b>${d.date}</b></div>
              <div>預測收盤：<b>${d.predicted_close?.toFixed(2)} TWD</b></div>
            `);
        })
        .on('mousemove', (event) => {
          tooltip.style('left', (event.pageX + 14) + 'px').style('top', (event.pageY - 28) + 'px');
        })
        .on('mouseout', () => tooltip.style('display', 'none'));
    });

  }, [data, predictions, modelVisibility]);

  return (
    <div ref={wrapRef} style={{ width: '100%', overflowX: 'auto' }}>
      <svg ref={svgRef} style={{ display: 'block' }} />
    </div>
  );
}

// ─── 成交量 Bar Chart（D3）──────────────────────────────────────────────────

function VolumeChart({ data }) {
  const svgRef = useRef(null);
  const wrapRef = useRef(null);

  useEffect(() => {
    if (!data.length || !svgRef.current || !wrapRef.current) return;

    const margin = { top: 10, right: 20, bottom: 50, left: 60 };
    const totalWidth = wrapRef.current.clientWidth || 800;
    const totalHeight = 200;
    const width = totalWidth - margin.left - margin.right;
    const height = totalHeight - margin.top - margin.bottom;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', totalWidth)
      .attr('height', totalHeight)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleBand()
      .domain(data.map(d => d.date))
      .range([0, width])
      .padding(0.2);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.volume) * 1.05])
      .range([height, 0]);

    const tickEvery = Math.ceil(data.length / 12);
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale)
        .tickValues(data.filter((_, i) => i % tickEvery === 0).map(d => d.date))
        .tickFormat(d => d.slice(5)))
      .call(g => g.select('.domain').attr('stroke', '#ddd'))
      .call(g => g.selectAll('text')
        .attr('transform', 'rotate(-40)')
        .style('text-anchor', 'end')
        .attr('dx', '-0.5em')
        .attr('dy', '0.15em')
        .attr('font-size', 10)
        .attr('fill', '#888'));

    svg.append('g')
      .call(d3.axisLeft(yScale).ticks(4).tickFormat(d => d >= 1000 ? `${(d / 1000).toFixed(0)}K` : d))
      .call(g => g.select('.domain').attr('stroke', '#ddd'))
      .call(g => g.selectAll('text').attr('font-size', 10).attr('fill', '#888'));

    svg.selectAll('.vol-bar')
      .data(data)
      .join('rect')
      .attr('class', 'vol-bar')
      .attr('x', d => xScale(d.date))
      .attr('y', d => yScale(d.volume || 0))
      .attr('width', xScale.bandwidth())
      .attr('height', d => Math.max(height - yScale(d.volume || 0), 0))
      .attr('fill', d => d.close >= d.open ? 'rgba(192,57,43,0.55)' : 'rgba(39,174,96,0.55)')
      .attr('rx', 1);

  }, [data]);

  return (
    <div ref={wrapRef} style={{ width: '100%', overflowX: 'auto' }}>
      <svg ref={svgRef} style={{ display: 'block' }} />
    </div>
  );
}

// ─── Model Toggle Panel ───────────────────────────────────────────────────────

function ModelTogglePanel({ modelVisibility, setModelVisibility, predictions, modelsLoaded }) {
  return (
    <div style={styles.modelPanel}>
      <div style={styles.modelPanelTitle}>AI 預測模型</div>
      {Object.entries(MODEL_CONFIG).map(([key, cfg]) => {
        const isLoaded = modelsLoaded.includes(key);
        const predCount = (predictions[key] || []).length;
        const enabled = modelVisibility[key];

        return (
          <label key={key} style={{ ...styles.modelToggle, opacity: isLoaded ? 1 : 0.45 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div
                onClick={() => isLoaded && setModelVisibility(v => ({ ...v, [key]: !v[key] }))}
                style={{
                  ...styles.checkbox,
                  background: enabled ? cfg.color : '#e5e7eb',
                  borderColor: enabled ? cfg.color : '#d1d5db',
                  cursor: isLoaded ? 'pointer' : 'not-allowed',
                }}
              >
                {enabled && <span style={styles.checkmark}>✓</span>}
              </div>

              <svg width="28" height="12">
                <line
                  x1="0" y1="6" x2="28" y2="6"
                  stroke={isLoaded ? cfg.color : '#ccc'}
                  strokeWidth="2"
                  strokeDasharray={cfg.dash}
                />
              </svg>

              <div>
                <div style={{ ...styles.modelName, color: enabled && isLoaded ? cfg.color : '#888' }}>
                  {cfg.label}
                </div>
                <div style={styles.modelMeta}>
                  {isLoaded ? `${predCount} 筆預測` : '模型未載入'}
                </div>
              </div>
            </div>
          </label>
        );
      })}
      {modelsLoaded.length === 0 && (
        <div style={{ fontSize: 11, color: '#aaa', marginTop: 8, lineHeight: 1.5 }}>
          將 .onnx 檔放到後端 models/ 資料夾並重啟伺服器以啟用預測
        </div>
      )}
    </div>
  );
}

// ─── Metric Card ──────────────────────────────────────────────────────────────

function MetricCard({ label, value, sub, color }) {
  return (
    <div style={styles.card}>
      <div style={styles.cardLabel}>{label}</div>
      <div style={{ ...styles.cardValue, color: color || '#1a1a1a' }}>{value}</div>
      {sub && <div style={styles.cardSub}>{sub}</div>}
    </div>
  );
}

// ─── Global Tooltip ──────────────────────────────────────────────────────────

function GlobalTooltip() {
  return (
    <div
      id="candle-tooltip"
      style={{
        display: 'none', position: 'fixed', zIndex: 9999,
        background: 'rgba(255,255,255,0.97)', border: '1px solid #e5e5e5',
        borderRadius: 10, padding: '10px 14px', fontSize: 13,
        boxShadow: '0 4px 20px rgba(0,0,0,0.12)', pointerEvents: 'none',
        lineHeight: 1.7, minWidth: 160,
      }}
    />
  );
}

// ─── Stock list ───────────────────────────────────────────────────────────────

const STOCK_LIST = [
  { no: '2330', name: '台積電' },
  { no: '2317', name: '鴻海' },
  { no: '2308', name: '台達電' },
  { no: '2454', name: '聯發科' },
  { no: '2882', name: '國泰金' },
  { no: '1301', name: '台塑' },
  { no: '2412', name: '中華電' },
  { no: '3008', name: '大立光' },
  { no: '2886', name: '兆豐金' },
  { no: '2891', name: '中信金' },
  { no: '1216', name: '統一' },
  { no: '2357', name: '華碩' },
];

const MONTH_OPTIONS = [1, 3];

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [stockNo, setStockNo] = useState('2308');
  const [months, setMonths] = useState(3);
  const [data, setData] = useState([]);
  const [predictions, setPredictions] = useState({});
  const [modelsLoaded, setModelsLoaded] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelVisibility, setModelVisibility] = useState({
    LSTM: true,
    GRU: true,
    BiLSTM: true,
    Transformer: true,
  });

  const load = useCallback(async (no, m) => {
    setLoading(true);
    setError(null);
    try {
      const json = await fetchStockData(no, m);
      setData(json.data || []);
      setPredictions(json.predictions || {});
      setModelsLoaded(json.models_loaded || []);
    } catch (e) {
      setError(e.message);
      setData([]);
      setPredictions({});
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(stockNo, months); }, [stockNo, months, load]);

  const last = data[data.length - 1] || {};
  const chg = last.daily_change_pct ?? 0;
  const chgColor = chg > 0 ? '#C0392B' : chg < 0 ? '#27AE60' : '#888';
  const chgStr = chg > 0 ? `▲ +${chg.toFixed(2)}%` : chg < 0 ? `▼ ${chg.toFixed(2)}%` : `${chg.toFixed(2)}%`;

  const visibleModelCount = Object.keys(MODEL_CONFIG).filter(k => modelVisibility[k] && modelsLoaded.includes(k)).length;

  return (
    <div style={styles.page}>
      <GlobalTooltip />

      <div style={styles.header}>
        <div style={styles.headerLeft}>
          <a
            href={`https://finance.yahoo.com/quote/${stockNo}.TW`}
            target="_blank"
            rel="noopener noreferrer"
            style={{ ...styles.badge, textDecoration: 'none', cursor: 'pointer' }}
          >
            Yahoo Finance ↗
          </a>
          <h1 style={styles.title}>
            {STOCK_LIST.find(s => s.no === stockNo)?.name ?? stockNo}
          </h1>
        </div>
        <div style={styles.searchRow}>
          <select
            style={styles.select}
            value={stockNo}
            onChange={e => setStockNo(e.target.value)}
          >
            {STOCK_LIST.map(s => (
              <option key={s.no} value={s.no}>{s.no} {s.name}</option>
            ))}
          </select>
          {MONTH_OPTIONS.map(m => (
            <button
              key={m}
              style={{ ...styles.tab, ...(months === m ? styles.tabActive : {}), marginLeft: 4 }}
              onClick={() => setMonths(m)}
            >
              {m}M
            </button>
          ))}
        </div>
      </div>

      {loading && (
        <div style={styles.modalOverlay}>
          <div style={styles.modalBox}>
            <div style={styles.spinner} />
            <div style={{ fontSize: 15, color: '#444', marginTop: 16 }}>資料載入中...</div>
            <div style={{ fontSize: 12, color: '#aaa', marginTop: 6 }}>正在取得 {stockNo} 資料及模型推論</div>
          </div>
        </div>
      )}
      {error && <div style={{ ...styles.status, color: '#C0392B' }}>錯誤：{error}</div>}

      {data.length > 0 && (
        <>
          <div style={styles.stockInfo}>
            <span style={styles.stockNo}>{stockNo}</span>
            <span style={{ color: chgColor, fontWeight: 700, fontSize: 20 }}>{chgStr}</span>
            <span style={{ color: '#888', fontSize: 13 }}>最新收盤 {last.close?.toFixed(2)} TWD</span>
          </div>

          <div style={styles.metrics}>
            <MetricCard label="收盤價" value={last.close?.toFixed(2) ?? '—'} sub="TWD" color={chgColor} />
            <MetricCard label="開盤價" value={last.open?.toFixed(2) ?? '—'} sub="TWD" />
            <MetricCard label="最高" value={last.high?.toFixed(2) ?? '—'} sub="TWD" color="#C0392B" />
            <MetricCard label="最低" value={last.low?.toFixed(2) ?? '—'} sub="TWD" color="#27AE60" />
            <MetricCard label="成交量" value={last.volume?.toLocaleString() ?? '—'} sub="張" />
            <MetricCard label="成交筆數" value={last.transactions?.toLocaleString() ?? '—'} sub="筆" />
          </div>

          <div style={styles.chartSection}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 20, flexWrap: 'wrap' }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={styles.chartTitle}>
                  K 線圖
                  <span style={styles.legend}>
                    <span style={{ color: '#C0392B' }}>■</span> 紅 K（上漲）
                    <span style={{ color: '#27AE60', marginLeft: 10 }}>■</span> 綠 K（下跌）
                    {visibleModelCount > 0 && Object.entries(MODEL_CONFIG).map(([key, cfg]) =>
                      modelVisibility[key] && modelsLoaded.includes(key) ? (
                        <span key={key} style={{ marginLeft: 10, color: cfg.color }}>— {cfg.label}</span>
                      ) : null
                    )}
                  </span>
                </div>
                <CandlestickChart
                  data={data}
                  predictions={predictions}
                  modelVisibility={modelVisibility}
                />
              </div>

              <ModelTogglePanel
                modelVisibility={modelVisibility}
                setModelVisibility={setModelVisibility}
                predictions={predictions}
                modelsLoaded={modelsLoaded}
              />
            </div>
          </div>

          <div style={styles.chartSection}>
            <div style={styles.chartTitle}>成交量（張）</div>
            <VolumeChart data={data} />
          </div>

          <div style={styles.tableSection}>
            <div style={{ ...styles.chartTitle, marginBottom: 12 }}>每日明細</div>
            <div style={styles.tableWrap}>
              <table style={styles.table}>
                <thead>
                  <tr>
                    {['日期', '股票', '開盤', '最高', '最低', '收盤', '成交量(張)', '成交筆數', '漲跌幅%'].map(h => (
                      <th key={h} style={styles.th}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[...data].reverse().map((row, i) => {
                    const pct = row.daily_change_pct ?? 0;
                    const pctColor = pct > 0 ? '#C0392B' : pct < 0 ? '#27AE60' : '#888';
                    return (
                      <tr key={row.date} style={i % 2 === 0 ? {} : { background: '#f8f9fa' }}>
                        <td style={styles.td}>{row.date}</td>
                        <td style={styles.td}>{row.symbol}</td>
                        <td style={styles.tdR}>{row.open?.toFixed(2) ?? '—'}</td>
                        <td style={{ ...styles.tdR, color: '#C0392B' }}>{row.high?.toFixed(2) ?? '—'}</td>
                        <td style={{ ...styles.tdR, color: '#27AE60' }}>{row.low?.toFixed(2) ?? '—'}</td>
                        <td style={{ ...styles.tdR, fontWeight: 700 }}>{row.close?.toFixed(2) ?? '—'}</td>
                        <td style={styles.tdR}>{row.volume?.toLocaleString() ?? '—'}</td>
                        <td style={styles.tdR}>{row.transactions?.toLocaleString() ?? '—'}</td>
                        <td style={{ ...styles.tdR, color: pctColor, fontWeight: 700 }}>
                          {pct > 0 ? `+${pct.toFixed(2)}%` : `${pct.toFixed(2)}%`}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ─── 樣式 ─────────────────────────────────────────────────────────────────────

const styles = {
  page: { maxWidth: 1200, margin: '0 auto', padding: '24px 20px', fontFamily: 'system-ui, sans-serif', color: '#1a1a1a' },
  header: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, marginBottom: 20 },
  headerLeft: { display: 'flex', alignItems: 'center', gap: 10 },
  badge: { background: '#E6F1FB', color: '#0C447C', fontSize: 12, fontWeight: 600, padding: '3px 10px', borderRadius: 99 },
  title: { fontSize: 22, fontWeight: 700, margin: 0 },
  searchRow: { display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' },
  select: { padding: '7px 12px', border: '1px solid #d0d0d0', borderRadius: 8, fontSize: 14, width: 200, outline: 'none', background: '#fff', cursor: 'pointer' },
  tab: { padding: '6px 14px', border: '1px solid #d0d0d0', borderRadius: 99, fontSize: 13, cursor: 'pointer', background: 'transparent', color: '#555' },
  tabActive: { background: '#f0f4f8', borderColor: '#aac4e6', color: '#185FA5', fontWeight: 600 },
  modalOverlay: { position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.35)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9998 },
  modalBox: { background: '#fff', borderRadius: 16, padding: '32px 40px', textAlign: 'center', boxShadow: '0 8px 32px rgba(0,0,0,0.18)', minWidth: 220 },
  spinner: { width: 36, height: 36, border: '3px solid #eee', borderTop: '3px solid #185FA5', borderRadius: '50%', animation: 'spin 0.8s linear infinite', margin: '0 auto' },
  status: { textAlign: 'center', padding: '2rem', fontSize: 14, color: '#888' },
  stockInfo: { display: 'flex', alignItems: 'center', gap: 14, marginBottom: 16 },
  stockNo: { fontSize: 26, fontWeight: 700 },
  metrics: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 10, marginBottom: 20 },
  card: { background: '#f5f6f7', borderRadius: 10, padding: '12px 16px' },
  cardLabel: { fontSize: 12, color: '#888', marginBottom: 4 },
  cardValue: { fontSize: 20, fontWeight: 600 },
  cardSub: { fontSize: 11, color: '#aaa', marginTop: 2 },
  chartSection: { background: '#fff', border: '1px solid #eee', borderRadius: 12, padding: '16px 20px', marginBottom: 16 },
  chartTitle: { fontSize: 14, fontWeight: 600, color: '#555', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' },
  legend: { fontSize: 12, color: '#888', fontWeight: 400 },
  tableSection: { marginBottom: 40 },
  tableWrap: { border: '1px solid #eee', borderRadius: 12, overflow: 'auto' },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
  th: { background: '#f5f6f7', padding: '10px 12px', textAlign: 'right', fontWeight: 600, fontSize: 12, color: '#666', whiteSpace: 'nowrap', borderBottom: '1px solid #eee' },
  td: { padding: '8px 12px', borderBottom: '1px solid #f0f0f0', color: '#555', whiteSpace: 'nowrap' },
  tdR: { padding: '8px 12px', borderBottom: '1px solid #f0f0f0', textAlign: 'right', whiteSpace: 'nowrap' },
  modelPanel: {
    minWidth: 170, maxWidth: 200,
    background: '#fafafa', border: '1px solid #eee',
    borderRadius: 10, padding: '14px 16px', flexShrink: 0,
  },
  modelPanelTitle: {
    fontSize: 12, fontWeight: 700, color: '#444',
    letterSpacing: '0.04em', textTransform: 'uppercase',
    marginBottom: 12, paddingBottom: 8, borderBottom: '1px solid #eee',
  },
  modelToggle: { display: 'block', marginBottom: 12, cursor: 'pointer', userSelect: 'none' },
  checkbox: {
    width: 18, height: 18, borderRadius: 4, border: '2px solid',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    transition: 'all 0.15s', flexShrink: 0,
  },
  checkmark: { color: '#fff', fontSize: 11, fontWeight: 700, lineHeight: 1 },
  modelName: { fontSize: 13, fontWeight: 600, lineHeight: 1.2 },
  modelMeta: { fontSize: 11, color: '#aaa', marginTop: 1 },
};