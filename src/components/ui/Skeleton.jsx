import React from "react";
import "./Skeleton.module.css";
const SkeletonElement = ({ type }) => {
    const classes = `skeleton ${type}`;
    return <div className={classes}></div>;
};

const Shimmer = () => {
    return (
        <div className="shimmer-wrapper">
            <div className="shimmer"></div>
        </div>
    );
};

export { Shimmer };
export default SkeletonElement;