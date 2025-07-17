import React from "react";
import {
    Home,
    Tag,
    Zap,
    CheckSquare,
    Settings, DownloadCloud, Cpu, Rocket,
} from "lucide-react";
import styles from "./Layout.module.css";
import { NavLink, useParams } from "react-router-dom";

const menuItems = [
    {label: "Home", icon: Home, path: ""},
    {label: "labeling", icon: Tag, path: "labeling"},
    {label: "Training", icon: Cpu, path: "training"},
    {label: "Optimization", icon: Zap, path: "optimization"},
    {label: "Validation", icon: CheckSquare, path: "validation"},
    {label: "Deployment", icon: Rocket, path: "deployment"},
    {label: "Service Process", icon: Settings, path: "service-process"},
];

export default function Sidebar() {
    const { projectId } = useParams();
    const basePath = `/projects/${projectId}`;
    return (
        <aside className={styles.sidebar}>
            <div className={styles["sidebar-inner"]}>
                <h1 className={styles["sidebar-project-name"]}>Project Name</h1>
                <nav className={styles["sidebar-nav"]} aria-label="Main navigation">
                    <ul>
                        {menuItems.map((item) => {
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
                </nav>
            </div>
        </aside>
    );
}
