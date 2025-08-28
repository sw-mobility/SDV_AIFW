import React from 'react';
import { X, TrendingUp, Clock, Target, BarChart3, Zap } from 'lucide-react';
import styles from './MetricsDetailModal.module.css';

/**
 * Validation Metrics 상세 모달
 * @param {Object} metrics - Validation metrics data
 * @param {boolean} isOpen - Modal open state
 * @param {Function} onClose - Close modal function
 * @returns {React.JSX.Element|null}
 */
const MetricsDetailModal = ({ metrics, isOpen, onClose }) => {
  if (!isOpen || !metrics) return null;

  const formatNumber = (num) => {
    if (typeof num !== 'number') return 'N/A';
    return num.toFixed(4);
  };

  // 디버깅을 위한 로그
  console.log('Metrics data in modal:', metrics);
  console.log('Available metrics keys:', Object.keys(metrics || {}));

  const formatSpeed = (speed) => {
    if (typeof speed !== 'number') return 'N/A';
    return `${speed.toFixed(2)}ms`;
  };

  const getClassAPColor = (ap) => {
    if (ap >= 0.8) return '#10b981'; // Green
    if (ap >= 0.6) return '#f59e0b'; // Yellow
    if (ap >= 0.4) return '#f97316'; // Orange
    return '#ef4444'; // Red
  };

  return (
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h2 className={styles.modalTitle}>
            <BarChart3 size={24} />
            Validation Metrics Details
          </h2>
          <button className={styles.closeButton} onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <div className={styles.modalBody}>
          {/* 주요 성능 지표 */}
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>
              <Target size={18} />
              Performance Metrics
            </h3>
            <div className={styles.metricsGrid}>
                             <div className={styles.metricCard}>
                 <div className={styles.metricLabel}>mAP@0.5</div>
                 <div className={styles.metricValue}>{formatNumber(metrics['mAP_0.5'] || metrics.mAP_0_5)}</div>
                 <div className={styles.metricDesc}>Mean Average Precision at IoU=0.5</div>
               </div>
               <div className={styles.metricCard}>
                 <div className={styles.metricLabel}>mAP@0.5:0.95</div>
                 <div className={styles.metricValue}>{formatNumber(metrics['mAP_0.5:0.95'] || metrics.mAP_0_5_0_95)}</div>
                 <div className={styles.metricDesc}>Mean Average Precision at IoU=0.5:0.95</div>
               </div>
              <div className={styles.metricCard}>
                <div className={styles.metricLabel}>Precision</div>
                <div className={styles.metricValue}>{formatNumber(metrics.mean_precision)}</div>
                <div className={styles.metricDesc}>Mean Precision</div>
              </div>
              <div className={styles.metricCard}>
                <div className={styles.metricLabel}>Recall</div>
                <div className={styles.metricValue}>{formatNumber(metrics.mean_recall)}</div>
                <div className={styles.metricDesc}>Mean Recall</div>
              </div>
            </div>
          </div>

          {/* 클래스별 AP */}
          {metrics.class_ap && metrics.class_names && (
            <div className={styles.section}>
              <h3 className={styles.sectionTitle}>
                <TrendingUp size={18} />
                Class-wise Average Precision
              </h3>
              <div className={styles.classAPGrid}>
                {Object.entries(metrics.class_ap).map(([classId, ap]) => (
                  <div key={classId} className={styles.classAPCard}>
                    <div className={styles.className}>
                      {metrics.class_names[classId] || `Class ${classId}`}
                    </div>
                    <div 
                      className={styles.classAPValue}
                      style={{ color: getClassAPColor(ap) }}
                    >
                      {formatNumber(ap)}
                    </div>
                    <div className={styles.classAPBar}>
                      <div 
                        className={styles.classAPBarFill}
                        style={{ 
                          width: `${ap * 100}%`,
                          backgroundColor: getClassAPColor(ap)
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 추론 속도 */}
          {metrics.inference_speed && (
            <div className={styles.section}>
              <h3 className={styles.sectionTitle}>
                <Zap size={18} />
                Inference Speed
              </h3>
              <div className={styles.speedGrid}>
                <div className={styles.speedCard}>
                  <div className={styles.speedLabel}>Preprocess</div>
                  <div className={styles.speedValue}>{formatSpeed(metrics.inference_speed.preprocess)}</div>
                </div>
                <div className={styles.speedCard}>
                  <div className={styles.speedLabel}>Inference</div>
                  <div className={styles.speedValue}>{formatSpeed(metrics.inference_speed.inference)}</div>
                </div>
                <div className={styles.speedCard}>
                  <div className={styles.speedLabel}>Postprocess</div>
                  <div className={styles.speedValue}>{formatSpeed(metrics.inference_speed.postprocess)}</div>
                </div>
                <div className={styles.speedCard}>
                  <div className={styles.speedLabel}>Loss</div>
                  <div className={styles.speedValue}>{formatNumber(metrics.inference_speed.loss)}</div>
                </div>
              </div>
            </div>
          )}

          {/* 요약 정보 */}
          <div className={styles.section}>
            <h3 className={styles.sectionTitle}>
              <Clock size={18} />
              Summary
            </h3>
                         <div className={styles.summaryGrid}>
               <div className={styles.summaryItem}>
                 <span className={styles.summaryLabel}>Total Classes:</span>
                 <span className={styles.summaryValue}>{metrics.total_classes || 'N/A'}</span>
               </div>
               <div className={styles.summaryItem}>
                 <span className={styles.summaryLabel}>Classes with AP:</span>
                 <span className={styles.summaryValue}>{metrics.total_classes_with_ap || 'N/A'}</span>
               </div>
               <div className={styles.summaryItem}>
                 <span className={styles.summaryLabel}>Validation Completed:</span>
                 <span className={styles.summaryValue}>
                   {metrics.validation_completed ? 'Yes' : 'No'}
                 </span>
               </div>
             </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MetricsDetailModal;
