import { FC, useState } from 'react';
import { useGetSimilarElements } from '../core/api';
import { useAppContext } from '../core/context';

interface Props {
  elementId: string;
  onSelectElement: (id: string) => void;
}

export const SimilarElements: FC<Props> = ({ elementId, onSelectElement }) => {
  const { appContext } = useAppContext();
  const projectSlug = appContext.currentProject?.params?.project_slug || null;
  const { getSimilarElements } = useGetSimilarElements(projectSlug);

  const [results, setResults] = useState<{ id: string; score: number }[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [show, setShow] = useState(false);

  // get ALL available features, not just embeddings_
  const allFeatures = Object.keys(appContext.currentProject?.features?.available || {});
  
  // try embeddings_ first, fallback to all features
  const availableFeatures = allFeatures.filter((f) => f.startsWith('embeddings_'));
  const featuresOptions = availableFeatures.length > 0 ? availableFeatures : allFeatures;

  const [selectedFeature, setSelectedFeature] = useState<string>('');
  const effectiveFeature = selectedFeature || featuresOptions[0] || '';

  const handleSearch = async () => {
    if (!effectiveFeature) return;
    setLoading(true);
    const res = await getSimilarElements(elementId, effectiveFeature, 10);
    setResults(res);
    setLoading(false);
    setShow(true);
  };

  // no features at all
  if (featuresOptions.length === 0) {
    return (
      <div className="similar-elements mt-2">
        <small className="text-muted">⚠️ No features computed. Compute embeddings first to use similarity search.</small>
      </div>
    );
  }

  return (
    <div className="similar-elements mt-2" style={{ width: '100%' }}>
      <div className="d-flex gap-2 align-items-center flex-wrap">
        <select
          className="form-select form-select-sm"
          style={{ width: 'auto', maxWidth: '200px' }}
          value={selectedFeature || featuresOptions[0]}
          onChange={(e) => setSelectedFeature(e.target.value)}
        >
          {featuresOptions.map((f) => (
            <option key={f} value={f}>{f}</option>
          ))}
        </select>
        <button
          className="btn btn-sm btn-outline-secondary"
          onClick={handleSearch}
          disabled={loading}
          title="Find semantically similar documents"
        >
          {loading ? '⏳' : '🔍'} Similar
        </button>
        {show && results && (
          <button className="btn btn-sm btn-link p-0" onClick={() => setShow(false)}>
            Hide
          </button>
        )}
      </div>

      {show && results && (
        <div className="similar-results mt-2 border rounded p-2" style={{ maxHeight: '200px', overflowY: 'auto', width: '100%' }}>
          {results.length === 0 ? (
            <small className="text-muted">No similar elements found</small>
          ) : (
            results.map((r) => (
              <div
                key={r.id}
                className="d-flex justify-content-between align-items-center py-1 border-bottom"
              >
                <button
                  className="btn btn-link btn-sm p-0 text-start"
                  onClick={() => onSelectElement(r.id)}
                  style={{ fontSize: '0.8rem' }}
                >
                  {r.id}
                </button>
                <span className="badge bg-secondary ms-2" style={{ fontSize: '0.7rem' }}>
                  {(r.score * 100).toFixed(1)}%
                </span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
};