import { useRef, useState, useEffect } from "react";
import { Checkbox, Panel, DefaultButton, TextField, SpinButton, Dropdown, IDropdownOption } from "@fluentui/react";
import { SparkleFilled } from "@fluentui/react-icons";
import readNDJSONStream from "ndjson-readablestream";

import styles from "./Chat.module.css";

import { chatApi, RetrievalMode, ChatAppResponse, ChatAppResponseOrError, ChatAppRequest, ResponseMessage } from "../../api";
import { Answer, AnswerError, AnswerLoading } from "../../components/Answer";
import { QuestionInput } from "../../components/QuestionInput";
import { ExampleList } from "../../components/Example";
import { UserChatMessage } from "../../components/UserChatMessage";
import { AnalysisPanel, AnalysisPanelTabs } from "../../components/AnalysisPanel";
import { SettingsButton } from "../../components/SettingsButton";
import { ClearChatButton } from "../../components/ClearChatButton";
import { SaveChatButton } from "../../components/SaveChatButton";
import { useLogin, getToken } from "../../authConfig";
import { useMsal } from "@azure/msal-react";
import { TokenClaimsDisplay } from "../../components/TokenClaimsDisplay";
import { HelperButtons } from "../../components/HelperButtons";
import { RequestType } from "../../api";
import { Button } from "react-bootstrap";

const Chat = () => {
    const [isConfigPanelOpen, setIsConfigPanelOpen] = useState(false);
    const [promptTemplate, setPromptTemplate] = useState<string>("");
    const [retrieveCount, setRetrieveCount] = useState<number>(3);
    const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>(RetrievalMode.Hybrid);
    const [useSemanticRanker, setUseSemanticRanker] = useState<boolean>(true);
    const [useSemanticCaptions, setUseSemanticCaptions] = useState<boolean>(false);
    const [excludeCategory, setExcludeCategory] = useState<string>("");
    const [useSuggestFollowupQuestions, setUseSuggestFollowupQuestions] = useState<boolean>(false);
    const [useOidSecurityFilter, setUseOidSecurityFilter] = useState<boolean>(false);
    const [useGroupsSecurityFilter, setUseGroupsSecurityFilter] = useState<boolean>(false);

    const lastQuestionRef = useRef<string>("");
    const chatMessageStreamEnd = useRef<HTMLDivElement | null>(null);

    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<unknown>();

    const [activeCitation, setActiveCitation] = useState<string>();
    const [activeAnalysisPanelTab, setActiveAnalysisPanelTab] = useState<AnalysisPanelTabs | undefined>(undefined);

    const [selectedAnswer, setSelectedAnswer] = useState<number>(0);
    const [answers, setAnswers] = useState<[user: string, response: string][]>([]);

    const handleAsyncRequest = async (question: string, answers: [string, ChatAppResponse][], setAnswers: Function, responseBody: ReadableStream<any>) => {
        let answer: string = "";
        let askResponse: ChatAppResponse = {} as ChatAppResponse;

        for await (const event of readNDJSONStream(responseBody)) {
            if (event["choices"] && event["choices"][0]["context"] && event["choices"][0]["context"]["data_points"]) {
                event["choices"][0]["message"] = event["choices"][0]["delta"];
                askResponse = event;
            } else if (event["choices"] && event["choices"][0]["delta"]["content"]) {
                setIsLoading(false);
            } else if (event["error"]) {
                throw Error(event["error"]);
            }
        }

        const fullResponse: ChatAppResponse = {
            ...askResponse,
            choices: [{ ...askResponse.choices[0], message: { content: answer, role: askResponse.choices[0].message.role } }]
        };
        return fullResponse;
    };

    const client = useLogin ? useMsal().instance : undefined;

    const makeApiRequest = async (question: string, requestType: RequestType = "question", requestContent: string = "placeholder") => {
        lastQuestionRef.current = question;

        error && setError(undefined);
        setIsLoading(true);
        setActiveCitation(undefined);
        setActiveAnalysisPanelTab(undefined);

        const token = client ? await getToken(client) : undefined;

        try {
            const messages: ResponseMessage[] = answers.flatMap(a => [
                { content: a[0], role: "user" },
                { content: a[1], role: "assistant" }
            ]);
            let prompt = "";
            let content: string = "";
            switch (requestType) {
                case "question":
                    prompt = "This is a question aside from the ongoing negotiation case, so please answer this question directly.";
                    content = question;
                    break;
                case "upload_background":
                    prompt =
                        'This is a piece of background information about the negotiation. I request that you wait until all background and additional information have been provided before generating the island of agreement. In the meantime, please respond with "Received the background information. You may proceed to provide more information or request to generate the Island of Agreement."';
                    content = prompt + requestContent;
                    break;
                case "upload_additional":
                    prompt =
                        'This is a piece of additional information about the negotiation. I request that you wait until all background and additional information have been provided before generating the island of agreement. In the meantime, please respond with "Received the additional information. You may proceed to provide more information or request to generate the Island of Agreement."';
                    content = prompt + requestContent;
                    break;
                case "generate_iog":
                    content =
                        "Based on the above definition about the island of agreement and the example, " +
                        "generate an Island of Agreement Table in tabular format which displays contested facts, agreed facts, " +
                        "convergent norms, and divergent norms among different stakeholders in the negotiation process. " +
                        "For each category, list the items in the order of importance.";
                    break;
            }
            console.log("!!content: " + content);

            const request: ChatAppRequest = {
                messages: [{ content: content, role: "user" }],
                context: {
                    overrides: {
                        prompt_template: promptTemplate.length === 0 ? undefined : promptTemplate,
                        exclude_category: excludeCategory.length === 0 ? undefined : excludeCategory,
                        top: retrieveCount,
                        retrieval_mode: retrievalMode,
                        semantic_ranker: useSemanticRanker,
                        semantic_captions: useSemanticCaptions,
                        suggest_followup_questions: useSuggestFollowupQuestions,
                        use_oid_security_filter: useOidSecurityFilter,
                        use_groups_security_filter: useGroupsSecurityFilter
                    }
                },
                session_state: null
            };
            console.log("!!request is:" + JSON.stringify(request));
            const response = await chatApi(request, token?.accessToken);
            if (!response.body) {
                throw Error("No response body");
            }
            // if (shouldStream) {
            //     const parsedResponse: ChatAppResponse = await handleAsyncRequest(question, answers, setAnswers, response.body);
            //     setAnswers([...answers, [question, parsedResponse]]);
            // } else {
            const parsedResponse: ChatAppResponseOrError = await response.json();
            if (response.status > 299 || !response.ok) {
                throw Error(parsedResponse.error || "Unknown error");
            }
            setAnswers([...answers, [question, parsedResponse.answer as string]]);
            console.log(JSON.stringify(answers));
            // }
        } catch (e) {
            setError(e);
        } finally {
            setIsLoading(false);
        }
    };

    const clearChat = () => {
        lastQuestionRef.current = "";
        error && setError(undefined);
        setActiveCitation(undefined);
        setActiveAnalysisPanelTab(undefined);
        setAnswers([]);
        setIsLoading(false);
    };

    const saveChat = () => {
        // save messages to backend server
        // bring to chat session history page
    };

    useEffect(() => chatMessageStreamEnd.current?.scrollIntoView({ behavior: "smooth" }), [isLoading]);
    // useEffect(() => chatMessageStreamEnd.current?.scrollIntoView({ behavior: "auto" }), [streamedAnswers]);

    const onPromptTemplateChange = (_ev?: React.FormEvent<HTMLInputElement | HTMLTextAreaElement>, newValue?: string) => {
        setPromptTemplate(newValue || "");
    };

    const onRetrieveCountChange = (_ev?: React.SyntheticEvent<HTMLElement, Event>, newValue?: string) => {
        setRetrieveCount(parseInt(newValue || "3"));
    };

    const onRetrievalModeChange = (_ev: React.FormEvent<HTMLDivElement>, option?: IDropdownOption<RetrievalMode> | undefined, index?: number | undefined) => {
        setRetrievalMode(option?.data || RetrievalMode.Hybrid);
    };

    const onUseSemanticRankerChange = (_ev?: React.FormEvent<HTMLElement | HTMLInputElement>, checked?: boolean) => {
        setUseSemanticRanker(!!checked);
    };

    const onUseSemanticCaptionsChange = (_ev?: React.FormEvent<HTMLElement | HTMLInputElement>, checked?: boolean) => {
        setUseSemanticCaptions(!!checked);
    };

    const onExcludeCategoryChanged = (_ev?: React.FormEvent, newValue?: string) => {
        setExcludeCategory(newValue || "");
    };

    const onUseSuggestFollowupQuestionsChange = (_ev?: React.FormEvent<HTMLElement | HTMLInputElement>, checked?: boolean) => {
        setUseSuggestFollowupQuestions(!!checked);
    };

    const onUseOidSecurityFilterChange = (_ev?: React.FormEvent<HTMLElement | HTMLInputElement>, checked?: boolean) => {
        setUseOidSecurityFilter(!!checked);
    };

    const onUseGroupsSecurityFilterChange = (_ev?: React.FormEvent<HTMLElement | HTMLInputElement>, checked?: boolean) => {
        setUseGroupsSecurityFilter(!!checked);
    };

    const onExampleClicked = (example: string) => {
        makeApiRequest(example);
    };

    const onShowCitation = (citation: string, index: number) => {
        if (activeCitation === citation && activeAnalysisPanelTab === AnalysisPanelTabs.CitationTab && selectedAnswer === index) {
            setActiveAnalysisPanelTab(undefined);
        } else {
            setActiveCitation(citation);
            setActiveAnalysisPanelTab(AnalysisPanelTabs.CitationTab);
        }

        setSelectedAnswer(index);
    };

    const onToggleTab = (tab: AnalysisPanelTabs, index: number) => {
        if (activeAnalysisPanelTab === tab && selectedAnswer === index) {
            setActiveAnalysisPanelTab(undefined);
        } else {
            setActiveAnalysisPanelTab(tab);
        }

        setSelectedAnswer(index);
    };

    return (
        <div className={styles.container}>
            <div className={styles.commandsContainer}>
                <ClearChatButton className={styles.commandButton} onClick={clearChat} disabled={!lastQuestionRef.current || isLoading} />
                <SaveChatButton className={styles.commandButton} onClick={saveChat} disabled={isLoading} />
                {/* Consider comment out this line */}
                {/* <SettingsButton className={styles.commandButton} onClick={() => setIsConfigPanelOpen(!isConfigPanelOpen)} />*/}
            </div>
            <div className={styles.chatRoot}>
                <div className={styles.chatContainer}>
                    {!lastQuestionRef.current ? (
                        <div className={styles.chatEmptyState}>
                            <SparkleFilled fontSize={"120px"} primaryFill={"rgba(115, 118, 225, 1)"} aria-hidden="true" aria-label="Chat logo" />
                            <h1 className={styles.chatEmptyStateTitle}>Talk to Your Website</h1>
                            <h2 className={styles.chatEmptyStateSubtitle}>Ask anything you want to know about your website.</h2>
                            <ExampleList onExampleClicked={onExampleClicked} />
                        </div>
                    ) : (
                        <div className={styles.chatMessageStream}>
                            {answers.map((answer, index) => (
                                <div key={index}>
                                    <UserChatMessage message={answer[0]} />
                                    <div className={styles.chatMessageGpt}>
                                        <Answer key={index} answer={answer[1]} isSelected={false} />
                                    </div>
                                </div>
                            ))}
                            {isLoading && (
                                <>
                                    <UserChatMessage message={lastQuestionRef.current} />
                                    <div className={styles.chatMessageGptMinWidth}>
                                        <AnswerLoading />
                                    </div>
                                </>
                            )}
                            {error ? (
                                <>
                                    <UserChatMessage message={lastQuestionRef.current} />
                                    <div className={styles.chatMessageGptMinWidth}>
                                        <AnswerError error={error.toString()} onRetry={() => makeApiRequest(lastQuestionRef.current)} />
                                    </div>
                                </>
                            ) : null}
                            <div ref={chatMessageStreamEnd} />
                        </div>
                    )}

                    <div className={styles.chatInput}>
                        {/* <HelperButtons onRequest={(question, requestType) => makeApiRequest(question, requestType)} /> */}
                        <QuestionInput
                            clearOnSend
                            placeholder="Type a new question (e.g. What is Harvard?)"
                            disabled={isLoading}
                            onSend={question => makeApiRequest(question)}
                        />
                    </div>
                </div>

                {/* {answers.length > 0 && activeAnalysisPanelTab && (
                    <AnalysisPanel
                        className={styles.chatAnalysisPanel}
                        activeCitation={activeCitation}
                        onActiveTabChanged={x => onToggleTab(x, selectedAnswer)}
                        citationHeight="810px"
                        answer={answers[selectedAnswer][1]}
                        activeTab={activeAnalysisPanelTab}
                    />
                )} */}

                <Panel
                    headerText="Configure answer generation"
                    isOpen={isConfigPanelOpen}
                    isBlocking={false}
                    onDismiss={() => setIsConfigPanelOpen(false)}
                    closeButtonAriaLabel="Close"
                    onRenderFooterContent={() => <DefaultButton onClick={() => setIsConfigPanelOpen(false)}>Close</DefaultButton>}
                    isFooterAtBottom={true}
                >
                    <TextField
                        className={styles.chatSettingsSeparator}
                        defaultValue={promptTemplate}
                        label="Override prompt template"
                        multiline
                        autoAdjustHeight
                        onChange={onPromptTemplateChange}
                    />

                    <SpinButton
                        className={styles.chatSettingsSeparator}
                        label="Retrieve this many search results:"
                        min={1}
                        max={50}
                        defaultValue={retrieveCount.toString()}
                        onChange={onRetrieveCountChange}
                    />
                    <TextField className={styles.chatSettingsSeparator} label="Exclude category" onChange={onExcludeCategoryChanged} />
                    <Checkbox
                        className={styles.chatSettingsSeparator}
                        checked={useSemanticRanker}
                        label="Use semantic ranker for retrieval"
                        onChange={onUseSemanticRankerChange}
                    />
                    <Checkbox
                        className={styles.chatSettingsSeparator}
                        checked={useSemanticCaptions}
                        label="Use query-contextual summaries instead of whole documents"
                        onChange={onUseSemanticCaptionsChange}
                        disabled={!useSemanticRanker}
                    />
                    <Checkbox
                        className={styles.chatSettingsSeparator}
                        checked={useSuggestFollowupQuestions}
                        label="Suggest follow-up questions"
                        onChange={onUseSuggestFollowupQuestionsChange}
                    />
                    {useLogin && (
                        <Checkbox
                            className={styles.chatSettingsSeparator}
                            checked={useOidSecurityFilter}
                            label="Use oid security filter"
                            disabled={!client?.getActiveAccount()}
                            onChange={onUseOidSecurityFilterChange}
                        />
                    )}
                    {useLogin && (
                        <Checkbox
                            className={styles.chatSettingsSeparator}
                            checked={useGroupsSecurityFilter}
                            label="Use groups security filter"
                            disabled={!client?.getActiveAccount()}
                            onChange={onUseGroupsSecurityFilterChange}
                        />
                    )}
                    <Dropdown
                        className={styles.chatSettingsSeparator}
                        label="Retrieval mode"
                        options={[
                            { key: "hybrid", text: "Vectors + Text (Hybrid)", selected: retrievalMode == RetrievalMode.Hybrid, data: RetrievalMode.Hybrid },
                            { key: "vectors", text: "Vectors", selected: retrievalMode == RetrievalMode.Vectors, data: RetrievalMode.Vectors },
                            { key: "text", text: "Text", selected: retrievalMode == RetrievalMode.Text, data: RetrievalMode.Text }
                        ]}
                        required
                        onChange={onRetrievalModeChange}
                    />
                    <Checkbox className={styles.chatSettingsSeparator} label="Stream chat completion responses" />
                    {useLogin && <TokenClaimsDisplay />}
                </Panel>
            </div>
        </div>
    );
};

export default Chat;
