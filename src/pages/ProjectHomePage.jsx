import React from "react";
import Table from "../components/ui/Table.jsx";
import ProgressBar from "../components/ui/ProgressBar.jsx";

const ProjectHomePage = () => {
    return (
        <>
            <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 28, fontWeight: 700, color: "#222" }}>Project Name</div>
                <div style={{ fontSize: 18, fontWeight: 500, color: "#444", marginTop: 18 }}>Dashboard</div>
            </div>
            <section style={{ margin: "20px 0 32px 0" }}>
                <div style={{ fontSize: 17, fontWeight: 600, color: "#222", marginBottom: 18 }}>
                    Process and Resource Monitoring
                </div>
                <div style={{ maxWidth: 420, marginBottom: 8 }}>
                    <ProgressBar label={null} percentage={70} />
                </div>
                <Table
                    columns={["Pod", "Status", "CPU", "Memory", "GPU"]}
                    data={[
                        ["Pod1", "Running", "80%", "90%", "90%"],
                        ["Pod2", "Idle", "10%", "20%", "15%"],
                        ["Pod3", "Running", "70%", "50%", "85%"],
                    ]}
                />
            </section>
            <section style={{ margin: "40px 0 32px 0" }}>
                <div style={{ fontSize: 17, fontWeight: 600, color: "#222", marginBottom: 18 }}>
                    Recent Dataset List
                </div>
                <Table
                    columns={["Dataset Name", "Version", "Size", "Last Modified"]}
                    data={[
                        ["Dataset A", "v1.2", "10GB", "2023-09-15"],
                        ["Dataset B", "v2.0", "5GB", "2023-09-10"],
                        ["Dataset C", "v1.5", "8GB", "2023-09-08"],
                    ]}
                />
            </section>
            <section style={{ margin: "40px 0 0 0" }}>
                <div style={{ fontSize: 17, fontWeight: 600, color: "#222", marginBottom: 18 }}>
                    Recently Modified Code Files
                </div>
                <Table
                    columns={["Filename", "Last Modified"]}
                    data={[
                        ["dataLoader.py", "2023-09-15"],
                        ["train.py", "2023-09-09"],
                        ["test.py", "2023-09-05"],
                    ]}
                />
            </section>
        </>
    );
};

export default ProjectHomePage;
