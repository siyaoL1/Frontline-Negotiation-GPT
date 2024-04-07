import React from "react";
import { Text } from "@fluentui/react/lib/Text";
import { IconButton } from "@fluentui/react/lib/Button";

import styles from "./NewAnswer.module.css"; // Assume CSS module for styling

interface NewAnswerProps {
    answer: string;
}

export const NewAnswer: React.FC<NewAnswerProps> = ({ answer }) => {
    return (
        <div className={styles.answerContainer}>
            <Text className={styles.answerText}>{answer}</Text>
        </div>
    );
};
