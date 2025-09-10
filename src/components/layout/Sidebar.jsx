import React, { useState, useEffect } from "react";
import {
    Home,
    Tag,
    Zap,
    CheckSquare,
    Settings, DownloadCloud, Cpu, Rocket, Code,
} from "lucide-react";
import styles from "./Layout.module.css";
import { NavLink, useParams } from "react-router-dom";
import { getProjectById } from "../../api/projects.js";
import { uid } from "../../api/uid.js";
import { SkeletonTitle } from "../ui/atoms/Skeleton.jsx";

const menuGroups = [
    {
        title: "Overview",
        items: [
            {label: "Home", icon: Home, path: ""},
        ]
    },
    {
        title: "Development",
        items: [
            {label: "labeling", icon: Tag, path: "labeling"},
            {label: "Code Editor", icon: Code, path: "editor"},
        ],
        hasSeparator: true
    },
    {
        title: "ML Pipeline",
        items: [
            {label: "Training", icon: Cpu, path: "training"},
            {label: "Optimization", icon: Zap, path: "optimization"},
            {label: "Validation", icon: CheckSquare, path: "validation"},
            {label: "Deployment", icon: Rocket, path: "deployment"},
            {label: "Service Process", icon: Settings, path: "service-process"},
        ]
    }
];

export default function Sidebar() {
    const { projectName } = useParams();
    const [projectData, setProjectData] = useState(null);
    const [loading, setLoading] = useState(true);
    const basePath = `/projects/${projectName}`;

    useEffect(() => {
        const fetchProjectData = async () => {
            try {
                setLoading(true);
                // projectName으로 project를 찾기 위해 모든 project를 가져와서 필터링
                const response = await fetch(`http://localhost:5002/projects/projects/`, {
                    headers: {
                        'uid': uid
                    }
                });
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
        <aside className={styles.sidebar}>
            <div className={styles["sidebar-inner"]}>
                <h1 className={styles["sidebar-project-name"]}>
                    {loading ? (
                        <SkeletonTitle 
                            width="200px" 
                            height="24px" 
                            className={styles.sidebarSkeleton}
                        />
                    ) : (
                        projectData?.name || 'Project Not Found'
                    )}
                </h1>
                <nav className={styles["sidebar-nav"]} aria-label="Main navigation">
                    {menuGroups.map((group, groupIndex) => (
                        <div key={groupIndex} className={styles["sidebar-group"]}>
                            {group.hasSeparator && groupIndex > 0 && (
                                <div className={styles["sidebar-separator"]}></div>
                            )}
                            <div className={styles["sidebar-group-title"]}>
                                {group.title}
                            </div>
                            <ul>
                                {group.items.map((item) => {
                                    const to = item.path ? `${basePath}/${item.path}` : basePath;
                                    const Icon = item.icon;
                                    return (
                                        <li key={item.label}>
                                            <NavLink
                                                to={to}
                                                className={({ isActive }) => [
                                                    styles["sidebar-nav-item"],
                                                    isActive ? styles["active"] : ""
                                                ].join(" ")}
                                                end={item.path === ""}
                                                style={{ textDecoration: 'none' }}
                                            >
                                                <Icon className={styles["sidebar-nav-icon"]} strokeWidth={1.5} />
                                                <span className={styles["sidebar-nav-label"]}>{item.label}</span>
                                            </NavLink>
                                        </li>
                                    );
                                })}
                            </ul>
                            {group.hasSeparator && groupIndex < menuGroups.length - 1 && (
                                <div className={styles["sidebar-separator"]}></div>
                            )}
                        </div>
                    ))}
                </nav>
            </div>
        </aside>
    );
}
