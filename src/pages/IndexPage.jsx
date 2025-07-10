import React from 'react';
import Header from './Header';
import Footer from './Footer';
import styles from '../components/layout/Layout.module.css'
import Card from "../components/ui/Card.jsx";

const IndexPage = () => (
    <div className={styles['main-layout']}>
        <Header />
        <div className={styles['main-body']}>
            <main className={styles['main-content']}>
                <Card/>
            </main>
        </div>
        <Footer />
    </div>
);

export default IndexPage;