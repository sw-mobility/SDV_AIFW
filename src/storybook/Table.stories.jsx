import Table from '../components/ui/Table.jsx';

export default {
  title: 'UI/Table',
  component: Table,
  parameters: {
    layout: 'centered',
  },
};

export const Default = {
  args: {
    columns: ['Name', 'Age', 'Email', 'Status'],
    data: [
      ['John Doe', '25', 'john@example.com', 'Active'],
      ['Jane Smith', '30', 'jane@example.com', 'Inactive'],
      ['Bob Johnson', '35', 'bob@example.com', 'Active'],
    ],
  },
};

export const PodStatus = {
  args: {
    columns: ['Pod', 'Status', 'CPU', 'Memory', 'GPU'],
    data: [
      ['Pod 1', 'Running', '80%', '60%', '90%'],
      ['Pod 2', 'Idle', '10%', '20%', '0%'],
      ['Pod 3', 'Running', '45%', '75%', '30%'],
    ],
  },
};

export const EmptyTable = {
  args: {
    columns: ['Name', 'Age', 'Email'],
    data: [],
  },
};

export const LargeTable = {
  args: {
    columns: ['ID', 'Product', 'Category', 'Price', 'Stock', 'Status'],
    data: [
      ['1', 'Laptop', 'Electronics', '$999', '50', 'In Stock'],
      ['2', 'Mouse', 'Electronics', '$25', '100', 'In Stock'],
      ['3', 'Keyboard', 'Electronics', '$75', '30', 'Low Stock'],
      ['4', 'Monitor', 'Electronics', '$299', '15', 'In Stock'],
      ['5', 'Headphones', 'Electronics', '$150', '0', 'Out of Stock'],
      ['6', 'Webcam', 'Electronics', '$80', '25', 'In Stock'],
    ],
  },
}; 