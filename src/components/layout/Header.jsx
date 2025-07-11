import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import styles from './Layout.module.css';
import DatasetDrawer from '../dataset/DatasetDrawer';
import Button from '@mui/material/Button';
import { Database } from 'lucide-react';

const Header = () => {
    const [drawerOpen, setDrawerOpen] = useState(false);
    const location = useLocation();

    const showDatasetButton = /^\/projects?(\/|$)/.test(location.pathname);

    return (
        <header className={styles.header}>
            <div className={styles['logo-container']}>
                <Link to="/" style={{ textDecoration: 'none' }}>
                    <img src="/logo.png" alt="KETI" className={styles.logo} />
                </Link>
            </div>
            <div style={{ flex: 1 }} />
            {showDatasetButton && (
                <Button
                    variant="outlined"
                    startIcon={<Database size={18} />}
                    onClick={() => setDrawerOpen(true)}
                    sx={{
                        fontWeight: 550,
                        textTransform: 'none',
                        borderColor : 'transparent',
                        boxShadow: 'none',
                        transition: 'all 0.2s',
                        '&:hover': {
                        },
                        mr: 0
                    }}
                >
                   Quick Data Management
                </Button>
            )}
            <DatasetDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} />
        </header>
    );
};

export default Header;
