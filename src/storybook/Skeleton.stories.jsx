import SkeletonElement, { Shimmer } from '../components/ui/Skeleton.jsx';

export default {
  title: 'UI/Skeleton',
  component: SkeletonElement,
  parameters: {
    layout: 'centered',
  },
  argTypes: {
    type: {
      control: { type: 'select' },
      options: ['text', 'title', 'avatar', 'thumbnail'],
    },
  },
};

export const Text = {
  args: {
    type: 'text',
  },
};

export const Title = {
  args: {
    type: 'title',
  },
};

export const Avatar = {
  args: {
    type: 'avatar',
  },
};

export const Thumbnail = {
  args: {
    type: 'thumbnail',
  },
};

export const AllTypes = {
  render: () => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', width: '300px' }}>
      <SkeletonElement type="title" />
      <SkeletonElement type="text" />
      <SkeletonElement type="text" />
      <SkeletonElement type="text" />
      <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
        <SkeletonElement type="avatar" />
        <div style={{ flex: 1 }}>
          <SkeletonElement type="text" />
          <SkeletonElement type="text" />
        </div>
      </div>
    </div>
  ),
};

export const CardSkeleton = {
  render: () => (
    <div style={{ 
      border: '1px solid #e0e0e0', 
      borderRadius: '8px', 
      padding: '20px', 
      width: '300px',
      position: 'relative',
      overflow: 'hidden'
    }}>
      <Shimmer />
      <SkeletonElement type="title" />
      <div style={{ marginTop: '10px' }}>
        <SkeletonElement type="text" />
        <SkeletonElement type="text" />
        <SkeletonElement type="text" />
      </div>
      <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
        <SkeletonElement type="thumbnail" />
        <SkeletonElement type="thumbnail" />
        <SkeletonElement type="thumbnail" />
      </div>
    </div>
  ),
}; 