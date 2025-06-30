import { useMemo, useState } from "react";
import { Stack, IconButton } from "@fluentui/react";
import DOMPurify from "dompurify";

import styles from "./Answer.module.css";

import { RAGChatCompletion } from "../../api/models";
import { parseAnswerToHtml } from "./AnswerParser";
import { AnswerIcon } from "./AnswerIcon";

interface Props {
    answer: RAGChatCompletion;
    isSelected?: boolean;
    isStreaming: boolean;
    onCitationClicked: (filePath: string) => void;
    onThoughtProcessClicked: () => void;
    onSupportingContentClicked: () => void;
    onFollowupQuestionClicked?: (question: string) => void;
    showFollowupQuestions?: boolean;
}

export const Answer = ({
    answer,
    isSelected,
    isStreaming,
    onCitationClicked,
    onThoughtProcessClicked,
    onSupportingContentClicked,
    onFollowupQuestionClicked,
    showFollowupQuestions
}: Props) => {
    const [isReferencesCollapsed, setIsReferencesCollapsed] = useState(true);
    const followupQuestions = answer.context.followup_questions;
    const messageContent = answer.message.content;
    const parsedAnswer = useMemo(() => parseAnswerToHtml(messageContent, isStreaming, onCitationClicked), [answer]);

    const sanitizedAnswerHtml = DOMPurify.sanitize(parsedAnswer.answerHtml);

    return (
        <Stack className={`${styles.answerContainer} ${isSelected && styles.selected}`} verticalAlign="space-between">
            <Stack.Item>
                <Stack horizontal horizontalAlign="space-between">
                    <AnswerIcon />
                    <div>
                        <IconButton
                            style={{ color: "black" }}
                            iconProps={{ iconName: "Lightbulb" }}
                            title="Show thought process"
                            ariaLabel="Show thought process"
                            onClick={() => onThoughtProcessClicked()}
                            disabled={!answer.context.thoughts?.length}
                        />
                    </div>
                </Stack>
            </Stack.Item>

            <Stack.Item grow>
                <div className={styles.answerText} dangerouslySetInnerHTML={{ __html: sanitizedAnswerHtml }}></div>
            </Stack.Item>

            {!!parsedAnswer.citations.length && (
                <Stack.Item>
                    <Stack horizontal wrap tokens={{ childrenGap: 5 }}>
                        <Stack horizontal verticalAlign="center" tokens={{ childrenGap: 5 }}>
                            <IconButton
                                iconProps={{ iconName: isReferencesCollapsed ? "ChevronDown" : "ChevronUp" }}
                                title={isReferencesCollapsed ? "Expand references" : "Collapse references"}
                                ariaLabel={isReferencesCollapsed ? "Expand references" : "Collapse references"}
                                onClick={() => setIsReferencesCollapsed(!isReferencesCollapsed)}
                            />
                            <span className={styles.citationLearnMore}>References:</span>
                        </Stack>
                    </Stack>
                    // 
                    {!isReferencesCollapsed && (
                        <ol>
                            {parsedAnswer.citations.map((rowId, ind) => {
                                // Obtenemos la fuente de datos. 'as any' nos da flexibilidad.
                                const source = answer.context.data_points[rowId] as any;
                                if (!source) return null;

                                // Comprobamos si la fuente es un Vehículo.
                                // La forma más segura es ver si tiene una propiedad única de los vehículos, como 'fabricante'.
                                const isVeiculo = source.hasOwnProperty('fabricante');

                                return (
                                    <li key={rowId}>
                                        {isVeiculo ? (
                                            // Si es un Vehículo, mostramos estos campos:
                                            <>
                                                <h4>Placa: {source.placa || "N/A"}</h4>
                                                <p className={styles.referenceMetadata}>Fabricante: {source.fabricante}</p>
                                                <p className={styles.referenceMetadata}>Año: {source.ano}</p>
                                                <p className={styles.referenceMetadata}>Tipo: {source.tipo_onibus}</p>
                                            </>
                                        ) : (
                                            // Si no, asumimos que es un Abastecimiento y mostramos estos otros:
                                            <>
                                                <h4>Abastecimiento (Placa: {source.placa || "N/A"})</h4>
                                                <p className={styles.referenceMetadata}>Fecha: {source.data}</p>
                                                <p className={styles.referenceMetadata}>Costo: {source.custo_combustivel}</p>
                                                <p className={styles.referenceMetadata}>Eficiencia: {source.km_diesel} km/l</p>
                                            </>
                                        )}
                                    </li>
                                );
                            })}
                        </ol>
                    )}     //                 
                </Stack.Item>
            )}

            {!!followupQuestions?.length && showFollowupQuestions && onFollowupQuestionClicked && (
                <Stack.Item>
                    <Stack horizontal wrap className={`${!!parsedAnswer.citations.length ? styles.followupQuestionsList : ""}`} tokens={{ childrenGap: 6 }}>
                        <span className={styles.followupQuestionLearnMore}>Follow-up questions:</span>
                        {followupQuestions.map((x, i) => {
                            return (
                                <a key={i} className={styles.followupQuestion} title={x} onClick={() => onFollowupQuestionClicked(x)}>
                                    {`${x}`}
                                </a>
                            );
                        })}
                    </Stack>
                </Stack.Item>
            )}
        </Stack>
    );
};
