import React from 'react';

interface RiskDashboardProps {
  assessment: string;
  confidence: number;
  keyDrivers: string[];
  metrics?: any;
  dataGaps: string[];
}

export default function RiskDashboard({ assessment, confidence, keyDrivers, metrics, dataGaps }: RiskDashboardProps) {
  return (
    <div style={{ display: 'grid', gap: '32px' }}>
      {/* Assessment Section */}
      <div>
        <span className="eyebrow" style={{ color: 'var(--primary)', marginBottom: '12px' }}>Assessment Status</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
          <div style={{ flex: 1, height: '4px', background: 'var(--canvas-soft)', borderRadius: '2px', overflow: 'hidden' }}>
            <div style={{ width: `${Math.round(confidence * 100)}%`, height: '100%', background: 'var(--primary)' }}></div>
          </div>
          <span className="tabular-numeric" style={{ fontSize: '13px', fontWeight: 400, color: 'var(--primary)' }}>
            {Math.round(confidence * 100)}% Confidence
          </span>
        </div>
        <p style={{ fontSize: '14px', lineHeight: '1.6', color: 'var(--ink-secondary)' }}>
          {assessment}
        </p>
      </div>

      {/* Metrics Table */}
      <div>
        <span className="eyebrow" style={{ marginBottom: '12px' }}>Transaction Metrics</span>
        <div style={{ display: 'grid', gap: '8px' }}>
          <MetricRow label="Purchase Price" value={metrics?.purchase_price} />
          <MetricRow label="Implied Entry" value={metrics?.implied_entry} />
          <MetricRow label="Liquidity Profile" value={metrics?.liquidity_profile} />
          <MetricRow label="NAV" value={metrics?.nav} />
          <MetricRow label="TVPI" value={metrics?.tvpi} />
        </div>
      </div>

      {/* Drivers & Gaps */}
      <div style={{ display: 'grid', gap: '24px' }}>
        {keyDrivers.length > 0 && (
          <div>
            <span className="eyebrow" style={{ marginBottom: '8px' }}>Key Drivers</span>
            <ul style={{ padding: 0, listStyle: 'none', display: 'grid', gap: '6px' }}>
              {keyDrivers.slice(0, 3).map((driver, i) => (
                <li key={i} style={{ fontSize: '13px', color: 'var(--ink-secondary)', display: 'flex', gap: '8px' }}>
                  <span style={{ color: 'var(--primary)' }}>•</span> {driver}
                </li>
              ))}
            </ul>
          </div>
        )}

        {dataGaps.length > 0 && (
          <div>
            <span className="eyebrow" style={{ color: '#ea2261', marginBottom: '8px' }}>Pending Analysis</span>
            <ul style={{ padding: 0, listStyle: 'none', display: 'grid', gap: '6px' }}>
              {dataGaps.slice(0, 3).map((gap, i) => (
                <li key={i} style={{ fontSize: '13px', color: 'var(--ink-mute)', display: 'flex', gap: '8px' }}>
                  <span style={{ color: '#ea2261' }}>!</span> {gap}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

function MetricRow({ label, value }: any) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 0', borderBottom: '1px solid var(--hairline)' }}>
      <span style={{ fontSize: '13px', color: 'var(--ink-mute)', fontWeight: 400 }}>{label}</span>
      <span className="tabular-numeric" style={{ fontSize: '14px', color: 'var(--ink)', fontWeight: 500 }}>{value || '—'}</span>
    </div>
  );
}
