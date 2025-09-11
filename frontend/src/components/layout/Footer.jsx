import React from 'react';
import styles from './Layout.module.css';

const Footer = () => {
    return (
        <footer className={styles.footer}>
            <div className={styles['footer-content']}>
                <span className={styles['footer-text']}>2025 © KETI. All rights reserved.</span>
            </div>
        </footer>
    );
};

export default Footer;