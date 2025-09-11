// 프로젝트/데이터셋 status 변환 유틸
export function getStatusColor(status) {
    switch (status) {
        case 'Active': return 'success';
        case 'Training': return 'warning';
        case 'Deployed': return 'info';
        case 'Processing': return 'warning';
        default: return 'default';
    }
}

export function getStatusText(status) {
    switch (status) {
        case 'Active': return 'active';
        case 'Training': return 'training';
        case 'Deployed': return 'deployed';
        case 'Processing': return 'processing';
        default: return status;
    }
} 