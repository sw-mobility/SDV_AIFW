import React, { useState, useEffect } from "react";
import Table from "../../shared/common/Table.jsx";
import styles from "./ProjectHomePage.module.css";
import { Box, Typography, Grid, Card, CardContent } from '@mui/material';
import { Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend
} from 'chart.js';
import { useParams } from 'react-router-dom';
import { uid } from '../../api/uid.js';

ChartJS.register(ArcElement, Tooltip, Legend);

const donutChartData = (label, value, color) => ({
  labels: [label, 'Unused'],
  datasets: [
    {
      data: [value, 100 - value],
      backgroundColor: [
        color,
        '#e5e7eb',
      ],
      borderWidth: 0,
      cutout: '70%',
      hoverOffset: 8,
    },
  ],
});

const donutChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { display: false },
    tooltip: { 
      enabled: true,
      position: 'nearest',
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      titleColor: '#fff',
      bodyColor: '#fff',
      borderColor: 'rgba(255, 255, 255, 0.1)',
      borderWidth: 1,
      cornerRadius: 8,
      displayColors: false,
      padding: 12,
    },
  },
  animation: {
    animateRotate: true,
    animateScale: true,
  },
  interaction: {
    intersect: false,
    mode: 'nearest',
  },
  layout: {
    padding: {
      top: 20,
      bottom: 20,
      left: 20,
      right: 20,
    },
  },
};

const podStatusData = {
  labels: ['Running', 'Idle', 'Failed'],
  datasets: [
    {
      data: [2, 1, 0],
      backgroundColor: ['#4f8cff', '#f59e42', '#ef4444'],
      borderWidth: 0,
    },
  ],
};
const podStatusOptions = {
  plugins: {
    legend: { display: true, position: 'bottom' },
    tooltip: { enabled: true },
  },
  cutout: '70%',
};

const recentAlerts = [
  { type: 'Warning', message: 'Pod2: Memory usage high', time: '2024-05-01 14:22' },
  { type: 'Error', message: 'Pod3: GPU not responding', time: '2024-05-01 13:58' },
];

const ProjectHomePage = () => {
  const { projectName } = useParams();
  const [projectData, setProjectData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchProjectData = async () => {
      try {
        setLoading(true);
        const response = await fetch(`http://localhost:5002/projects/projects/?uid=${encodeURIComponent(uid)}`);
        if (response.ok) {
          const data = await response.json();
          const project = data.find(p => p.name === projectName);
          if (project) {
            setProjectData(project);
          }
        }
      } catch (error) {
        console.error('Failed to fetch project data:', error);
      } finally {
        setLoading(false);
      }
    };

    if (projectName) {
      fetchProjectData();
    }
  }, [projectName]);

  return (
    <Box sx={{ maxWidth: 1200, margin: '0 auto', padding: { xs: '0 0', md: '0 0' }, background: 'var(--color-bg-primary)', boxSizing: 'border-box', mt: '-24px' }}>
      {/* Hero Section */}
      <div className={styles.hero}>
        <div className={styles.heroBlur} />
        <div className={styles.heroContent}>
          <div className={styles.heroTitle}>
            {loading ? 'Loading...' : (projectData?.name || 'Project Not Found')}
          </div>
          <div className={styles.heroDesc}>Dashboard</div>
        </div>
      </div>
      <Grid container spacing={3} justifyContent="center">
        {/* Resource Utilization */}
        <Grid column={{ xs: 12, md: 10 }}>
          <div className={styles.card}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 24 }}>
              <div>
                <div className={styles.cardTitle}>Resource Utilization</div>
                <div className={styles.cardSub}>Real-time usage of your cluster resources</div>
              </div>
              <div className={styles.donutWrap}>
                {[
                  { label: 'CPU', value: 70, color: '#4f8cff' },
                  { label: 'Memory', value: 55, color: '#22c55e' },
                  { label: 'GPU', value: 87, color: '#f59e42' },
                ].map((res) => (
                  <div key={res.label} className={styles.donutItem}>
                    <div className={styles.donutChart}>
                      <Doughnut data={donutChartData(res.label, res.value, res.color)} options={donutChartOptions} />
                      <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', pointerEvents: 'none' }}>
                        <span className={styles.donutValue} style={{ color: res.color }}>{res.value}%</span>
                        <span className={styles.donutLabel}>{res.label}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Grid>
        {/* Pod Status & Alerts */}
        <Grid column={{ xs: 12, md: 5 }}>
          <div className={styles.card}>
            <div className={styles.cardTitle}>Pod Status Distribution</div>
            <Box sx={{ width: 180, mx: 'auto' }}>
              <Doughnut data={podStatusData} options={podStatusOptions} />
            </Box>
          </div>
          <div className={styles.card}>
            <div className={styles.cardTitle}>Recent Alerts & Errors</div>
            <Box>
              {recentAlerts.length === 0 ? (
                <Typography color="text.secondary">No recent alerts.</Typography>
              ) : (
                recentAlerts.map((alert, idx) => (
                  <Box key={idx} sx={{ mb: 1, p: 1.2, borderRadius: 1.5, background: alert.type === 'Error' ? '#fef2f2' : '#fef9c3', color: alert.type === 'Error' ? '#dc2626' : '#b45309', fontWeight: 500, fontSize: 14 }}>
                    [{alert.type}] {alert.message} <span style={{ float: 'right', color: '#64748b', fontWeight: 400 }}>{alert.time}</span>
                  </Box>
                ))
              )}
            </Box>
          </div>
        </Grid>
        {/* Dataset & Code Tables */}
        <Grid column={{ xs: 12, md: 5 }}>
          <div className={styles.card}>
            <div className={styles.cardTitle}>Recent Dataset List</div>
            <Table
              columns={["Dataset Name", "Version", "Size", "Last Modified"]}
              data={[
                ["Dataset A", "v1.2", "10GB", "2023-09-15"],
                ["Dataset B", "v2.0", "5GB", "2023-09-10"],
                ["Dataset C", "v1.5", "8GB", "2023-09-08"],
              ]}
            />
          </div>
          <div className={styles.card}>
            <div className={styles.cardTitle}>Recently Modified Code Files</div>
            <Table
              columns={["Filename", "Last Modified"]}
              data={[
                ["dataLoader.py", "2023-09-15"],
                ["train.py", "2023-09-09"],
                ["test.py", "2023-09-05"],
              ]}
            />
          </div>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ProjectHomePage;
