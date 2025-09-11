import React from "react";
import styles from "./Skeleton.module.css";

/**
 * skeleton ui 컴포넌트
 * loading 시 animation 이 포함된 회색 필드를 넣어 ux 를 고려하는 방식
 * @param type
 * @param width
 * @param height
 * @param className
 * @returns {Element}
 * @constructor
 */
const SkeletonElement = ({ type, width, height, className }) => {
    const classes = `${styles.skeleton} ${styles[type]} ${className || ''}`;
    const style = {
        width: width,
        height: height
    };
    return <div className={classes} style={style}></div>;
};

const SkeletonText = ({ lines = 1, width = "100%", height = "16px", className }) => {
    return (
        <div className={className}>
            {Array(lines).fill(null).map((_, index) => (
                <SkeletonElement 
                    key={index}
                    type="text" 
                    width={width} 
                    height={height}
                    className={styles.skeletonText}
                />
            ))}
        </div>
    );
};

const SkeletonTitle = ({ width = "300px", height = "48px", className }) => {
    return (
        <SkeletonElement 
            type="title" 
            width={width} 
            height={height}
            className={className}
        />
    );
};

const SkeletonCard = ({ className }) => {
    return (
        <div className={`${styles.skeletonCard} ${className || ''}`}>
            <div className={styles.skeletonCardContent}>
                <div className={styles.skeletonStatus}></div>
                <div className={styles.skeletonIcon}></div>
                <div className={styles.skeletonName}></div>
                <div className={styles.skeletonDescription}></div>
                <div className={styles.skeletonDate}></div>
                <div className={styles.skeletonActions}>
                    <div className={styles.skeletonActionButton}></div>
                    <div className={styles.skeletonActionButton}></div>
                </div>
            </div>
        </div>
    );
};

const Shimmer = () => {
    return (
        <div className="shimmer-wrapper">
            <div className="shimmer"></div>
        </div>
    );
};

export { Shimmer, SkeletonText, SkeletonTitle, SkeletonCard };
export default SkeletonElement;