import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Modal, Button, Spinner, ProgressBar, Alert, Form, Row, Col } from 'react-bootstrap';
import { useWaxSuggest } from '../../core/api';
import { WaxParams } from '../../types';

interface Props {
  show: boolean;
  onHide: () => void;
  projectSlug: string;
  scheme: string;
  dataset?: string;
}

interface WaxResult {
  wax_distance: number;
  feature_relevance: number[];
  elements_id: string[];
  elements_text: string[];
  scores: number[];
  n_labeled: number;
  n_unlabeled: number;
}

const DEFAULT_PARAMS: WaxParams = {
  model: 'all-MiniLM-L6-v2',
  p: 2, q: 2, alpha: 2, beta: 2,
  n: 100, reg: 0.01, r: 4, C: 3,
  lr: 0.01, n_iter: 200, n_suggest: 10,
};

const POLL_INTERVAL_MS = 2000;

function getTopK(n: number): number {
  if (n < 50) return n;
  if (n < 200) return 20;
  if (n < 1000) return 30;
  return 50;
}

type ModalView = 'form' | 'running' | 'done' | 'error';

const PARAM_FIELDS: { key: keyof WaxParams; label: string; desc: string }[] = [
  { key: 'p',label: 'P',desc: 'Source distribution power' },
  { key: 'q',label: 'Q',desc: 'Target distribution power' },
  { key: 'alpha',label: 'Alpha',desc: 'Relevance exponent' },
  { key: 'beta',label: 'Beta',desc: 'Regularisation exponent' },
  { key: 'n',label: 'N',desc: 'OT iterations' },
  { key: 'reg',label: 'Reg',desc: 'Regularisation strength' },
  // {/* key: 'r',label: 'R',desc: 'U-WaX rank' */},
  // { /*key: 'C',label: 'C',desc: 'U-WaX clusters' */},
  // { /*key: 'lr',label: 'LR',desc: 'Learning rate' */},
  // { /*key: 'n_iter',label: 'N iter', desc: 'U-WaX iterations' },
  // { /*key: 'n_suggest', label: 'Top-N',  desc: 'Elements to suggest' },
];

export const WaxAnalysisModal: React.FC<Props> = ({
  show,
  onHide,
  projectSlug,
  scheme,
  dataset = 'train',
}) => {
  const { start, poll } = useWaxSuggest();
  const [view, setView]     = useState<ModalView>('form');
  const [params, setParams] = useState<WaxParams>(DEFAULT_PARAMS);
  const [result, setResult] = useState<WaxResult | null>(null);
  const [error, setError]   = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);

  const pollRef    = useRef<ReturnType<typeof setInterval> | null>(null);
  const timerRef   = useRef<ReturnType<typeof setInterval> | null>(null);
  const taskIdRef  = useRef<string | null>(null);

  // reset to form each time modal opens
  useEffect(() => {
    if (show) {
      setView('form');
      setResult(null);
      setError(null);
      setElapsed(0);
    }
  }, [show]);

  const stopPolling = useCallback(() => {
    if (pollRef.current)  clearInterval(pollRef.current);
    if (timerRef.current) clearInterval(timerRef.current);
  }, []);

  const handleClose = useCallback(() => {
    stopPolling();
    onHide();
  }, [stopPolling, onHide]);

  const handleParamChange = useCallback((key: keyof WaxParams, value: string) => {
    setParams((prev) => ({
      ...prev,
      [key]: typeof DEFAULT_PARAMS[key] === 'number' ? Number(value) : value,
    }));
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!scheme) return;
    setView('running');
    setElapsed(0);

    timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);

    const data = await start(projectSlug, scheme, dataset, params);
    if (!data) {
      stopPolling();
      setError('Failed to start WAX task');
      setView('error');
      return;
    }

    taskIdRef.current = data.task_id;
    pollRef.current = setInterval(async () => {
      const currenttaskId=taskIdRef.current;
      if(!currenttaskId) return;
      const res = await poll(currenttaskId);
      if (!res) return;
      if (res.status === 'done') {
        stopPolling();
        setResult(res.result);
        setView('done');
      } else if (res.status === 'error') {
        stopPolling();
        setError(res.error ?? 'Unknown error');
        setView('error');
      }
    }, POLL_INTERVAL_MS);
  }, [scheme, projectSlug, dataset, params, start, poll, stopPolling]);
  const topK = result ? getTopK(result.n_unlabeled) : 0;
  const maxScore = result ? Math.max(...result.scores) : 1;
  return (
    <Modal show={show} onHide={handleClose} size="lg" scrollable>
      <Modal.Header closeButton>
        <Modal.Title className="d-flex align-items-center gap-2">
          WAX Analysis
          {view !== 'form' && (
            <Button
              variant="link"
              size="sm"
              className="p-0 ms-1"
              onClick={() => { stopPolling(); setView('form'); }}
            >
              ← back to params
            </Button>
          )}
        </Modal.Title>
      </Modal.Header>

      <Modal.Body>
        {view === 'form' && (
          <Form>
            <Form.Group className="mb-3">
              <Form.Label><strong>Embedding model</strong></Form.Label>
              <Form.Control
                type="text"
                value={params.model}
                onChange={(e) => handleParamChange('model', e.target.value)}
              />
            </Form.Group>

            <Row>
              {PARAM_FIELDS.map(({ key, label, desc }) => (
                <Col xs={6} md={4} key={key} className="mb-3">
                  <Form.Label>
                    <strong>{label}</strong>{' '}
                    <small className="text-muted">{desc}</small>
                  </Form.Label>
                  <Form.Control
                    type="number"
                    value={params[key]}
                    step={key === 'reg' || key === 'lr' ? 0.001 : 1}
                    min={0}
                    onChange={(e) => handleParamChange(key, e.target.value)}
                  />
                </Col>
              ))}
            </Row>
          </Form>
        )}
        {view === 'running' && (
          <div className="text-center py-5">
            <Spinner animation="border" className="mb-3" />
            <p className="text-muted mb-3">Running WAX… {elapsed}s elapsed</p>
            <ProgressBar animated now={100} variant="info" />
          </div>
        )}
        {view === 'error' && (
          <Alert variant="danger">
            <strong>WAX failed:</strong> {error}
          </Alert>
        )}
        {view === 'done' && result && (
          <div>

            {/* summary stats */}
            <div className="d-flex gap-4 mb-4">
              <div>
                <small className="text-muted d-block">Wasserstein Distance</small>
                <h4 className="mb-0">{result.wax_distance.toFixed(4)}</h4>
              </div>
              <div>
                <small className="text-muted d-block">Labeled</small>
                <h4 className="mb-0">{result.n_labeled}</h4>
              </div>
              <div>
                <small className="text-muted d-block">Unlabeled</small>
                <h4 className="mb-0">{result.n_unlabeled}</h4>
              </div>
            </div>
            <h5>Top 10 suggested elements</h5>
            <div style={{ overflowX: 'auto' }} className="mb-4">
              <table className="table table-sm table-hover table-bordered align-middle">
                <thead className="table-light">
                  <tr>
                    <th style={{ width: '5%' }}>#</th>
                    <th style={{ width: '18%' }}>ID</th>
                    <th style={{ width: '52%' }}>Text</th>
                    <th style={{ width: '25%' }}>Score</th>
                  </tr>
                </thead>
                <tbody>
                  {result.elements_id.slice(0, topK).map((id, i) => (
                    <tr key={id}>
                      <td className="text-muted text-center">{i + 1}</td>
                      <td>
                        <code style={{ fontSize: '0.72rem', wordBreak: 'break-all' }}>
                          {id}
                        </code>
                      </td>
                      <td>
                        <div
                          style={{
                            maxHeight: '60px',
                            overflowY: 'auto',
                            fontSize: '0.85rem',
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word',
                          }}
                          title={result.elements_text[i]}
                        >
                          {result.elements_text[i]}
                        </div>
                      </td>
                      <td>
                        <div className="d-flex align-items-center gap-2">
                          <div
                            style={{
                              width: `${Math.round((result.scores[i] / maxScore) * 60)}px`,
                              minWidth: '4px',
                              height: '8px',
                              backgroundColor: '#4e79a7',
                              borderRadius: '4px',
                              flexShrink: 0,
                            }}
                          />
                          <span style={{ fontSize: '0.8rem', whiteSpace: 'nowrap' }}>
                            {result.scores[i].toFixed(4)}
                          </span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <h5>Feature relevance</h5>
          </div>
        )}
      </Modal.Body>

      <Modal.Footer>
        {view === 'form' && (
          <>
            <Button variant="secondary" onClick={handleClose}>
              Cancel
            </Button>
            <Button variant="primary" onClick={handleSubmit} disabled={!scheme}>
              Run WAX
            </Button>
          </>
        )}
        {view === 'running' && (
          <Button variant="danger" onClick={() => { stopPolling(); setView('form'); }}>
            Cancel
          </Button>
        )}
        {(view === 'done' || view === 'error') && (
          <>
            <Button variant="secondary" onClick={handleClose}>
              Close
            </Button>
            <Button variant="outline-primary" onClick={() => { stopPolling(); setView('form'); }}>
              Run again
            </Button>
          </>
        )}
      </Modal.Footer>
    </Modal>
  );
};