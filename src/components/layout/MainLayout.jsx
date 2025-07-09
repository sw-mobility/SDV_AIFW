import React from 'react';
import Header from './Header';
import Sidebar from './Sidebar';
import Footer from './Footer';
import styles from './Layout.module.css';
import { Outlet } from 'react-router-dom';

const MainLayout = () => (
  <div className={styles['main-layout']}>
    <Header />
    <div className={styles['main-body']}>
      <Sidebar />
      <main className={styles['main-content']}>
        <Outlet />
      </main>
    </div>
    <Footer />
  </div>
);

export default MainLayout; 