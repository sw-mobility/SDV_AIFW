import React, { useState} from "react";
import SectionTitle from "../components/ui/SectionTitle.jsx";

const ProjectHomePage = () => {
    const [title, setTitle] = useState("");
    return (
        <div>
            <SectionTitle children={title} size="lg" />
        </div>
    );
};

export default ProjectHomePage;
