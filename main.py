import os
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel


load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = os.getenv("OPENAI_MODEL")


class RequestBody(BaseModel):
    question: str
    answer: str
    role: str | None
    interview_type: str | None


def openai_prompt(client, model, req: RequestBody):
    question = (req.question,)
    answer = req.answer
    role = req.role if req.role else "None"
    interview_type = (
        ",".join(req.interview_type) if req.interview_type else "None"
    )
    input = "The candidate answers {answer} to the question {question}. The role is {role}. The interview type is {interview_type}".format(
        answer=answer,
        question=question,
        role=role,
        interview_type=interview_type,
    )
    response = client.responses.create(
        model=model,
        instructions="You are an expert interview assistant that generates insightful follow-up questions based on a candidate's response. Your goal is to help interviewers probe deeper into the candidate's experience, thought process, and qualifications. NEVER generate questions about age, disability, race, color, religion or belief, sex, national origin, gender, family status, marital status, health records, arrest records. Use gender-neutral language. Focus on job-related competencies only and avoid assumptions about background, experience, and/or circumstances. Maintain professional tone and respect. Generate a response that only contains the question and nothing else.",
        input=input,
    )
    return response


@app.post("/interview/generate-followups", status_code=status.HTTP_200_OK)
async def generate_question(req: RequestBody):
    if not req.question:
        raise HTTPException(
            status_code=400, detail="Question is not in request body."
        )
    if not req.answer:
        raise HTTPException(
            status_code=400, detail="Answer is not in request body."
        )
    openai_response = openai_prompt(client=client, model=model, req=req)
    # if openai_response.output and openai_response.output.content:
    if (
        openai_response
        and openai_response.output
        and openai_response.output[0].content
    ):
        result = openai_response.output[0].content[0].text
        return {
            "result": "success",
            "message": "Follow-up question generated.",
            "data": {"followup_question": result},
        }
    return raiseHTTPException(status_code=500, detail="Unexpected Error.")
