# 1단계: 빌드 환경
FROM node:20-alpine AS builder
WORKDIR /workspace5/frontend
COPY package.json ./
COPY package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

# 2단계: nginx로 정적 파일 서빙
FROM nginx:alpine
COPY --from=builder /workspace5/frontend/dist /usr/share/nginx/html
COPY --from=builder /workspace5/frontend/public /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]