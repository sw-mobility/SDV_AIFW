import React from 'react';
import { CardGrid } from './Card.jsx';
import styles from '../../pages/index_page/IndexPage.module.css';
import { ChevronDown } from 'lucide-react';

export default function ShowMoreGrid({ children, cardsPerPage = 8, showMore, onToggleShowMore }) {
    const visibleChildren = showMore ? children : children.slice(0, cardsPerPage);
    const remaining = children.length - cardsPerPage;
    return (
        <>
            <CardGrid>
                {visibleChildren}
            </CardGrid>
            {children.length > cardsPerPage && (
                <div className={styles.loadMoreContainer}>
                    <button onClick={onToggleShowMore} className={styles.moreButton}>
                        <span className={styles.moreText}>
                            {showMore
                                ? 'Show Less'
                                : `Show ${remaining} More`
                            }
                        </span>
                        <div className={styles.chevron + ' ' + (showMore ? styles.chevronUp : '')}>
                            <ChevronDown size={14} />
                        </div>
                    </button>
                </div>
            )}
        </>
    );
} 