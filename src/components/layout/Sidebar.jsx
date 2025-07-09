import React from "react";
import {
    Home,
    Target,
    Tag,
    Zap,
    CheckSquare,
    Database,
    Download,
    Cog,
    Settings,
} from "lucide-react";
import styles from "./Layout.module.css";

const menuItems = [
    {label: "Home", icon: Home, path: "/"},
    {label: "Training", icon: Target, path: "/training"},
    {label: "Labelling", icon: Tag, path: "/labelling"},
    {label: "Optimization", icon: Zap, path: "/optimization"},
    {label: "Validation", icon: CheckSquare, path: "/validation"},
    {label: "Data Management", icon: Database, path: "/data-management"},
    {label: "Data Acquisition", icon: Download, path: "/data-acquisition"},
    {label: "Deployment", icon: Cog, path: "/deployment"},
    {label: "Service Process", icon: Settings, path: "/service-process"},
];

function NavItem({icon: Icon, label}) {
    return (
        <li className={styles["sidebar-nav-item"]}>
            <Icon className={styles["sidebar-nav-icon"]} strokeWidth={1.5}/>
            <span className={styles["sidebar-nav-label"]}>{label}</span>
        </li>
    );
}

export default function Sidebar() {
    return (
        <aside className={styles.sidebar}>
            <div className={styles["sidebar-inner"]}>
                <h1 className={styles["sidebar-project-name"]}>Project Name</h1>

                <nav className={styles["sidebar-nav"]} aria-label="Main navigation">
                    <ul>
                        {menuItems.map((item) => (
                            <NavItem key={item.label} {...item} />
                        ))}
                    </ul>
                </nav>
            </div>
        </aside>
    );
}
