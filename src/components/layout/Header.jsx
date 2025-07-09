import React from 'react';
import styles from './Layout.module.css';

const Header = () => {
    return (
        <header className={styles.header}>
            <div className={styles['logo-container']}>
                <img src="/logo.png" alt="KETI" className={styles.logo} />
            </div>
        </header>
    );
};

export default Header;
