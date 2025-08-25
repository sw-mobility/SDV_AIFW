# 1단계: 빌드 환경
FROM node:20-alpine AS builder
WORKDIR /app
COPY ../../Downloads/Telegram%20Desktop/KETIAIFW_Train_Val_Opt_v0.1%20 package-lock.json* ./
RUN npm install
COPY ../../Downloads/Telegram%20Desktop/KETIAIFW_Train_Val_Opt_v0.1%20 .
RUN npm run build

# 2단계: nginx로 정적 파일 서빙
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY --from=builder /app/public /usr/share/nginx/html
COPY ../../Downloads/Telegram%20Desktop/KETIAIFW_Train_Val_Opt_v0.1%20 /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]