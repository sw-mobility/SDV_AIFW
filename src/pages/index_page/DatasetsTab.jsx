import React, { useState } from 'react';
import { Upload, Database, Tag, ChevronDown } from 'lucide-react';
import { useLocalStorageState } from '../../hooks/useLocalStorageState.js';
import Card, { CardGrid } from '../../components/ui/Card.jsx';
import styles from './IndexPage.module.css';
import Chip from '@mui/material/Chip';
import { Calendar, Download, Trash2 } from 'lucide-react';

const DatasetsTab = () => {
    const [showMore, setShowMore] = useState(false);
    const [dataType, setDataType] = useState('raw');
    const cardsPerPage = 8;

    const [rawDatasets, setRawDatasets] = useLocalStorageState('rawDatasets', [
        { id: 1, name: 'Image Dataset 1', type: 'Image', size: '2.3GB', lastModified: '2024-01-15', status: 'Active' },
        { id: 2, name: 'Image Dataset 2', type: 'Image', size: '1.8GB', lastModified: '2024-01-14', status: 'Active' },
        { id: 3, name: 'Text Dataset 1', type: 'Text', size: '500MB', lastModified: '2024-01-13', status: 'Active' },
        { id: 4, name: 'Audio Dataset 1', type: 'Audio', size: '3.2GB', lastModified: '2024-01-12', status: 'Active' },
        { id: 5, name: 'Video Dataset 1', type: 'Video', size: '5.1GB', lastModified: '2024-01-11', status: 'Active' },
    ]);

    const [labeledDatasets, setLabeledDatasets] = useLocalStorageState('labeledDatasets', [
        { id: 1, name: 'Labeled Image Dataset 1', type: 'Image', size: '2.8GB', lastModified: '2024-01-15', status: 'Active', labelCount: 15000 },
        { id: 2, name: 'Labeled Text Dataset 1', type: 'Text', size: '1.5GB', lastModified: '2024-01-14', status: 'Active', labelCount: 8000 },
        { id: 3, name: 'Labeled Audio Dataset 1', type: 'Audio', size: '3.5GB', lastModified: '2024-01-13', status: 'Active', labelCount: 12000 },
        { id: 4, name: 'Labeled Video Dataset 1', type: 'Video', size: '6.2GB', lastModified: '2024-01-12', status: 'Active', labelCount: 5000 },
        { id: 5, name: 'Labeled Image Dataset 2', type: 'Image', size: '1.9GB', lastModified: '2024-01-11', status: 'Active', labelCount: 9500 },
    ]);

    const handleToggleShowMore = () => {
        setShowMore(!showMore);
    };

    const handleDownload = (dataset) => {
        console.log('Downloading dataset:', dataset.name);
        // TODO: Implement download logic
    };

    const getStatusColor = (status) => {
        switch (status) {
            case 'Active': return 'success';
            case 'Training': return 'warning';
            case 'Deployed': return 'info';
            case 'Processing': return 'warning';
            default: return 'default';
        }
    };

    const getStatusText = (status) => {
        switch (status) {
            case 'Active': return 'active';
            case 'Training': return 'training';
            case 'Deployed': return 'deployed';
            case 'Processing': return 'processing';
            default: return status;
        }
    };

    const handleDelete = (dataset, isRaw) => {
        if (isRaw) {
            setRawDatasets(prev => prev.filter(d => d.id !== dataset.id));
        } else {
            setLabeledDatasets(prev => prev.filter(d => d.id !== dataset.id));
        }
    };

    const CreateDatasetCard = () => (
        <Card className={styles.createCard}>
            <div className={styles.createCardContent}>
                <Upload size={32} className={styles.createCardIcon} />
                <div className={styles.createCardText}>
                    Upload New Dataset
                </div>
            </div>
        </Card>
    );

    const RawDatasetCard = ({ dataset }) => (
        <Card className={styles.datasetCard}>
            <div className={styles.cardContent}>
                <Chip
                    label={getStatusText(dataset.status)}
                    color={getStatusColor(dataset.status)}
                    size="small"
                    variant="outlined"
                    className={styles.statusChip}
                />

                <div className={styles.cardIcon}>
                    <Database size={18} color="var(--color-text-secondary)" />
                </div>

                <div className={styles.cardName}>
                    {dataset.name}
                </div>

                <div className={styles.cardInfo}>
                    <span className={styles.cardType}>{dataset.type}</span>
                    <span className={styles.cardSize}>{dataset.size}</span>
                </div>

                {/* Raw data용 빈 공간 (labeled data의 labelCount와 동일한 높이) */}
                <div style={{ height: '24px' }}></div>

                <div className={styles.cardDate}>
                    <Calendar size={14} />
                    {dataset.lastModified}
                </div>

                <div className={styles.cardActions}>
                    <button className={styles.actionButton} title="Download" onClick={() => handleDownload(dataset)}>
                        <Download size={14} />
                    </button>
                    <button className={styles.actionButton} title="Delete" onClick={() => handleDelete(dataset, true)}>
                        <Trash2 size={14} />
                    </button>
                </div>
            </div>
        </Card>
    );

    const LabeledDatasetCard = ({ dataset }) => (
        <Card className={styles.datasetCard}>
            <div className={styles.cardContent}>
                <Chip
                    label={getStatusText(dataset.status)}
                    color={getStatusColor(dataset.status)}
                    size="small"
                    variant="outlined"
                    className={styles.statusChip}
                />

                <div className={styles.cardIcon}>
                    <Tag size={18} color="var(--color-text-secondary)" />
                </div>

                <div className={styles.cardName}>
                    {dataset.name}
                </div>

                <div className={styles.cardInfo}>
                    <span className={styles.cardType}>{dataset.type}</span>
                    <span className={styles.cardSize}>{dataset.size}</span>
                </div>

                {/* Labeled data용 labelCount 영역 */}
                <div className={styles.cardLabelCount}>
                    <Tag size={14} />
                    {dataset.labelCount.toLocaleString()} labels
                </div>

                <div className={styles.cardDate}>
                    <Calendar size={14} />
                    {dataset.lastModified}
                </div>

                <div className={styles.cardActions}>
                    <button className={styles.actionButton} title="Download" onClick={() => handleDownload(dataset)}>
                        <Download size={14} />
                    </button>
                    <button className={styles.actionButton} title="Delete" onClick={() => handleDelete(dataset, false)}>
                        <Trash2 size={14} />
                    </button>
                </div>
            </div>
        </Card>
    );

    const currentDatasets = dataType === 'raw' ? rawDatasets : labeledDatasets;
    const allDatasetCards = [
        <CreateDatasetCard key="create-dataset" />,
        ...currentDatasets.map(dataset => (
            dataType === 'raw' ? 
            <RawDatasetCard key={dataset.id} dataset={dataset} /> :
            <LabeledDatasetCard key={dataset.id} dataset={dataset} />
        ))
    ];
    const visibleDatasetCards = showMore ? allDatasetCards : allDatasetCards.slice(0, cardsPerPage);

    return (
        <>
            <div className={styles.dataTypeToggle}>
                <button
                    className={`${styles.dataTypeButton} ${dataType === 'raw' ? styles.activeDataType : ''}`}
                    onClick={() => setDataType('raw')}
                >
                    <Database size={16} />
                    Raw Data
                </button>
                <button
                    className={`${styles.dataTypeButton} ${dataType === 'labeled' ? styles.activeDataType : ''}`}
                    onClick={() => setDataType('labeled')}
                >
                    <Tag size={16} />
                    Labeled Data
                </button>
            </div>

            <CardGrid gap="2rem">
                {visibleDatasetCards}
            </CardGrid>

            {allDatasetCards.length > cardsPerPage && (
                <div className={styles.loadMoreContainer}>
                    <button
                        onClick={handleToggleShowMore}
                        className={styles.moreButton}
                    >
                        <span className={styles.moreText}>
                            {showMore ? 'Show Less' : `Show ${allDatasetCards.length - cardsPerPage} More`}
                        </span>
                        <div className={`${styles.chevron} ${showMore ? styles.chevronUp : ''}`}>
                            <ChevronDown size={14} />
                        </div>
                    </button>
                </div>
            )}
        </>
    );
};

export default DatasetsTab; 